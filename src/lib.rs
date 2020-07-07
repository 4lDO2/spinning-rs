#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

use core::cell::UnsafeCell;
use core::mem::{self, MaybeUninit};
use core::ptr;
use core::sync::atomic::{self, AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// An extremely simple spinlock, locking by compare-and-swapping a single flag and repeating.
pub struct RawMutex {
    locked: AtomicBool,
}

impl RawMutex {
    pub const fn const_new() -> Self {
        RawMutex {
            locked: AtomicBool::new(false),
        }
    }
}

unsafe impl lock_api::RawMutex for RawMutex {
    const INIT: Self = RawMutex::const_new();

    type GuardMarker = lock_api::GuardSend;

    fn lock(&self) {
        while !self.try_lock() {
            atomic::spin_loop_hint();
        }
    }

    fn try_lock(&self) -> bool {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    unsafe fn unlock(&self) {
        let prev = self.locked.fetch_and(false, Ordering::Release);
        debug_assert!(prev, "unlocking when not locked");
    }
}

#[derive(Debug)]
pub struct RawRwLock {
    // The state of the rwlock is composed as follows:
    //
    // 1. The base bits, all the way up to ARCH_POINTER_WIDTH - 4, are used to count the number of
    //    current locks. These also include the one occasional intent or write lock, and the reason
    //    for this is to allow for more simple atomic operations, since x86 has no instrucion doing
    //    both bit tests and addition.
    // 2. RWLOCK_STATE_ACTIVE_WRITER_BIT indicates that the lock currently holds a writer. When
    //    acquiring a shared lock, the counter may be incremented arbitrarily, but acquiring a
    //    shared lock must always fail when this bit is set.
    // 3. RWLOCK_STATE_ACTIVE_INTENT_BIT denotes that the rwlock currently holds an intent lock.
    //    Attaining a shared lock does ignores this bit, since intent locks only conflict with write
    //    locks, until they are upgraded to write locks.
    // 4. RWLOCK_STATE_PENDING_WRITER_BIT can be set at any time by a failed attempt at obtaining a
    //    write lock, due to shared locks already being present. In order to prevent write lock
    //    starvation, new shared locks cannot be acquired if this bit is set.
    state: AtomicUsize,
}
impl Clone for RawRwLock {
    fn clone(&self) -> Self {
        Self {
            state: AtomicUsize::new(0),
        }
    }
}

const RWLOCK_STATE_ACTIVE_WRITER_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 1);
const RWLOCK_STATE_ACTIVE_INTENT_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 2);

const RWLOCK_STATE_PENDING_WRITER_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 3);

const RWLOCK_STATE_EXTRA_MASK: usize = RWLOCK_STATE_ACTIVE_WRITER_BIT
    | RWLOCK_STATE_ACTIVE_INTENT_BIT
    | RWLOCK_STATE_PENDING_WRITER_BIT;
const RWLOCK_STATE_COUNT_MASK: usize = !RWLOCK_STATE_EXTRA_MASK;

impl RawRwLock {
    pub const fn const_new() -> Self {
        RawRwLock {
            state: AtomicUsize::new(0),
        }
    }

    fn try_lock_exclusive_raw(&self) -> (bool, bool) {
        let prev_state = self
            .state
            .fetch_or(RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::AcqRel);
        let current_state = prev_state | RWLOCK_STATE_PENDING_WRITER_BIT;
        let was_previously_pending = prev_state & RWLOCK_STATE_PENDING_WRITER_BIT != 0;

        if prev_state & RWLOCK_STATE_ACTIVE_INTENT_BIT != 0 {
            debug_assert_eq!(prev_state & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "simultaneously active INTENT and exclusive locks during exclusive lock acquisition");
            return (false, was_previously_pending);
        }
        if prev_state & RWLOCK_STATE_ACTIVE_WRITER_BIT != 0 {
            debug_assert_eq!(prev_state & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "simultaneously active intent and EXCLUSIVE locks during exclusive lock acquisition");
            return (false, was_previously_pending);
        }

        let success = self
            .state
            .compare_exchange(
                current_state,
                (current_state + 1) | RWLOCK_STATE_ACTIVE_WRITER_BIT,
                // make sure that the lock is updated to be an exclusive lock, before any
                // subsequent operations assume that underlying data to be readable (success
                // ordering).
                Ordering::Acquire,

                // a failed attempt to attain an exclusive lock means that the lock cannot later be
                // used for data accesses, and thus no synchronization is needed here (failure
                // ordering).
                Ordering::Relaxed,
            )
            .is_ok();
        (success, was_previously_pending)
    }
    unsafe fn try_upgrade_raw(&self) -> (bool, bool) {
        let prev = self
            .state
            .fetch_or(RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::Relaxed);

        debug_assert_ne!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "upgrading an intent lock into an exclusive lock when no intent lock was held"
        );

        let was_previously_pending = prev & RWLOCK_STATE_PENDING_WRITER_BIT != 0;

        let prev = self.state.compare_exchange_weak(
            RWLOCK_STATE_ACTIVE_INTENT_BIT | RWLOCK_STATE_PENDING_WRITER_BIT | 1,
            RWLOCK_STATE_ACTIVE_WRITER_BIT | 1,
            // make sure that the no operations that rely on the lock being an exclusive lock,
            // happen *after* the lock has been marked exclusive (success ordering).
            Ordering::Acquire,
            // no synchronization needs to happen, since a lock is neither obtained nor released
            // (failure ordering).
            Ordering::Relaxed,
        );

        if let Ok(prev_raw) = prev {
            debug_assert_eq!(
                prev_raw & RWLOCK_STATE_ACTIVE_WRITER_BIT,
                0,
                "upgrading an intent lock into an exclusive lock when an exclusive lock was held"
            );
        }

        (prev.is_ok(), was_previously_pending)
    }
}

unsafe impl lock_api::RawRwLock for RawRwLock {
    const INIT: Self = RawRwLock::const_new();

    type GuardMarker = lock_api::GuardSend;

    fn lock_shared(&self) {
        while !self.try_lock_shared() {
            atomic::spin_loop_hint();
        }
    }
    fn try_lock_shared(&self) -> bool {
        let prev = self.state.fetch_add(1, Ordering::Acquire);

        if prev & RWLOCK_STATE_PENDING_WRITER_BIT != 0 {
            // don't starve writers; writers are prioritized over readers
            return false;
        }

        if prev & RWLOCK_STATE_ACTIVE_WRITER_BIT != 0 {
            let new_prev = self.state.fetch_sub(1, Ordering::Release);
            debug_assert_ne!(
                new_prev & !(RWLOCK_STATE_ACTIVE_WRITER_BIT | RWLOCK_STATE_ACTIVE_INTENT_BIT),
                0,
                "overflow when subtracting rwlock counter"
            );
            return false;
        }
        true
    }
    fn lock_exclusive(&self) {
        while !self.try_lock_exclusive() {
            atomic::spin_loop_hint();
        }
    }
    fn try_lock_exclusive(&self) -> bool {
        let (success, was_previously_pending) = self.try_lock_exclusive_raw();

        if success && !was_previously_pending {
            self.state
                .fetch_and(!RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::Relaxed);
        }

        success
    }

    // releases a shared lock
    unsafe fn unlock_shared(&self) {
        let prev = self.state.fetch_sub(1, Ordering::Release);
        debug_assert_ne!(
            prev & RWLOCK_STATE_COUNT_MASK,
            0,
            "corrupted state flags because of subtraction overflow, when release a shared lock"
        );
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "releasing a shared lock while an exclusive lock was held"
        );
    }
    // releases an exclusive lock
    unsafe fn unlock_exclusive(&self) {
        let prev = self
            .state
            .fetch_sub(RWLOCK_STATE_ACTIVE_WRITER_BIT | 1, Ordering::Release);
        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "corrupted state flags because an exclusive lock release was tried when an exclusive lock was not held");
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "releasing an exclusive lock when an intent lock was held"
        );
    }
}
unsafe impl lock_api::RawRwLockDowngrade for RawRwLock {
    // downgrades an exclusive lock to a shared lock
    unsafe fn downgrade(&self) {
        let prev = self
            .state
            .fetch_and(!RWLOCK_STATE_ACTIVE_WRITER_BIT, Ordering::Release);
        debug_assert_ne!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "downgrading an exclusive lock to a shared lock when no exclusive lock was held"
        );
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "downgrading a exclusive lock to a shared lock when an intent lock was held"
        );
    }
}
unsafe impl lock_api::RawRwLockUpgrade for RawRwLock {
    // acquires an intent lock
    fn lock_upgradable(&self) {
        while !self.try_lock_upgradable() {
            atomic::spin_loop_hint();
        }
    }
    // tries to acquire an intent lock
    fn try_lock_upgradable(&self) -> bool {
        use lock_api::RawRwLock as _;

        // Begin by acquiring a shared lock.
        if !self.try_lock_shared() {
            return false;
        };

        // At this stage we know that it is completely impossible for a exclusive lock to exist, since
        // try_lock_shared() would return false in that case. Hence, all we have to do is setting
        // the active intent bit, and returning false if it was already set.
        let prev = self
            .state
            .fetch_or(RWLOCK_STATE_ACTIVE_INTENT_BIT, Ordering::AcqRel);
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "acquiring an intent lock while an exclusive lock was held"
        );

        prev & RWLOCK_STATE_ACTIVE_INTENT_BIT == 0
    }
    // releases an intent lock
    unsafe fn unlock_upgradable(&self) {
        // assumes that the lock is properly managed by lock_api; if RWLOCK_STATE_ACTIVE_INTENT_BIT
        // is not set and this method is called, the CPU will arithmetically borrow the bits below,
        // potentially corrupting the rwlock state entirely.
        let prev = self
            .state
            .fetch_sub(RWLOCK_STATE_ACTIVE_INTENT_BIT | 1, Ordering::Release);
        debug_assert_ne!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "releasing an intent lock while no intent lock was held"
        );
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "releasing an intent lock while an exclusive lock was held"
        );
    }
    // upgrades an intent lock into an exclusive lock
    unsafe fn upgrade(&self) {
        while !self.try_upgrade() {
            atomic::spin_loop_hint();
        }
    }

    // tries to upgrade an intent lock into an exclusive lock
    unsafe fn try_upgrade(&self) -> bool {
        let (success, was_previously_pending) = self.try_upgrade_raw();

        if success && !was_previously_pending {
            self.state
                .fetch_and(!RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::Relaxed);
        }

        success
    }
}
unsafe impl lock_api::RawRwLockUpgradeDowngrade for RawRwLock {
    // downgrades an exclusive lock to an intent lock
    unsafe fn downgrade_to_upgradable(&self) {
        let prev = self.state.fetch_xor(
            RWLOCK_STATE_ACTIVE_WRITER_BIT | RWLOCK_STATE_ACTIVE_INTENT_BIT,

            // ensure that previous memory accesses, which may have been writes (as we are
            // downgrading an exclusive lock), happen before other threads see this downgrade.
            Ordering::Release,
        );
        debug_assert_ne!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "downgrading a exclusive lock to an intent lock when no exclusive lock was held"
        );
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "downgrading a exclusive lock to an intent lock when an intent lock was held"
        );
    }
    // downgrades an intent lock into a shared lock
    unsafe fn downgrade_upgradable(&self) {
        let prev = self
            .state
            .fetch_and(
                !RWLOCK_STATE_ACTIVE_INTENT_BIT,

                // this ordering is correct, because no memory accesses that may have happened
                // before, or are going to happen afterwards, will have different rules regarding
                // mutability (intent locks are only shared locks but with the exception that there
                // can only be one intent lock at a time).
                Ordering::Relaxed,
            );
        debug_assert_eq!(
            prev & RWLOCK_STATE_ACTIVE_WRITER_BIT,
            0,
            "downgrading an intent lock while a exclusive lock was held"
        );
        debug_assert_ne!(
            prev & RWLOCK_STATE_ACTIVE_INTENT_BIT,
            0,
            "downgrading an intent lock where no intent lock was held"
        );
    }
}

pub type Mutex<T> = lock_api::Mutex<RawMutex, T>;
pub type MutexGuard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;
pub type MappedMutexGuard<'a, T> = lock_api::MappedMutexGuard<'a, RawMutex, T>;
pub type RwLock<T> = lock_api::RwLock<RawRwLock, T>;
pub type RwLockReadGuard<'a, T> = lock_api::RwLockReadGuard<'a, RawRwLock, T>;
pub type RwLockWriteGuard<'a, T> = lock_api::RwLockWriteGuard<'a, RawRwLock, T>;
pub type RwLockUpgradableReadGuard<'a, T> = lock_api::RwLockUpgradableReadGuard<'a, RawRwLock, T>;
pub type MappedRwLockReadGuard<'a, T> = lock_api::MappedRwLockReadGuard<'a, RawRwLock, T>;
pub type MappedRwLockWriteGuard<'a, T> = lock_api::MappedRwLockWriteGuard<'a, RawRwLock, T>;
pub type ReentrantMutex<T, G> = lock_api::ReentrantMutex<RawRwLock, G, T>;
pub type ReentrantMutexGuard<'a, T, G> = lock_api::ReentrantMutexGuard<'a, RawRwLock, G, T>;

/// A synchronization primitive which initializes a value lazily, once. Since this also includes a
/// value, it is a bit more like `Once` from `parking_lot` or `spin`.
pub struct Once<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Drop for Once<T> {
    fn drop(&mut self) {
        // we do not have to do any complex state manipulation here, since a mutable reference
        // guarantees that only there is an exclusive borrow to this struct.
        if *self.state.get_mut() != OnceState::Initialized as u8 {
            // nothing to drop
            return;
        }
        unsafe { ptr::drop_in_place(self.value.get() as *mut T) }
    }
}

unsafe impl<T: Send + Sync> Send for Once<T> {}
unsafe impl<T: Send + Sync> Sync for Once<T> {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OnceState {
    Uninitialized = 0,
    Initializing = 1,
    Initialized = 2,
}

impl<T> Once<T> {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(OnceState::Uninitialized as u8),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
    pub const fn uninitialized() -> Self {
        Self::new()
    }
    pub const fn initialized(value: T) -> Self {
        Self {
            state: AtomicU8::new(OnceState::Initialized as u8),
            value: UnsafeCell::new(MaybeUninit::new(value)),
        }
    }
    pub fn initialize(&self, value: T) -> Result<(), T> {
        match self.state.compare_exchange(
            OnceState::Uninitialized as u8,
            OnceState::Initializing as u8,

            // make sure that the actual pointer store must happen after other threads have seen
            // the updated state (success ordering).
            Ordering::Acquire,

            // if the once was not uninitialized, there will not be any additional stores, and
            // since we just return upon failure, Relaxed suffices here.
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                unsafe { ptr::write(self.value.get(), MaybeUninit::new(value)) };
                let old = self
                    .state
                    .swap(OnceState::Initialized as u8, Ordering::Release);
                debug_assert_eq!(
                    old,
                    OnceState::Initializing as u8,
                    "once state was modified when setting state to \"initialized\""
                );
                Ok(())
            }
            Err(_) => Err(value),
        }
    }
    pub fn try_call_once<'a, F>(&'a self, init: F) -> Result<&'a T, F>
    where
        F: FnOnce() -> T,
    {
        match self.state.compare_exchange(
            OnceState::Uninitialized as u8,
            OnceState::Initializing as u8,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => unsafe {
                ptr::write(self.value.get(), MaybeUninit::new(init()));
                let old = self
                    .state
                    .swap(OnceState::Initialized as u8, Ordering::Release);
                debug_assert_eq!(
                    old,
                    OnceState::Initializing as u8,
                    "once state was modified when setting state to \"initialized\""
                );
                Ok(&*(self.value.get() as *const T))
            },
            Err(other_state) if other_state == OnceState::Initialized as u8 => unsafe {
                Ok(&*(self.value.get() as *const T))
            },

            #[cfg(debug_assertions)]
            Err(other_state) if other_state == OnceState::Initializing as u8 => Err(init),

            #[cfg(debug_assertions)]
            Err(_) => unreachable!(),

            #[cfg(not(debug_assertions))]
            Err(_) => Err(init),
        }
    }
    pub fn call_once<'a, F>(&'a self, mut init: F) -> &'a T
    where
        F: FnOnce() -> T,
    {
        loop {
            match self.try_call_once(init) {
                Ok(reference) => return reference,
                Err(init_again) => {
                    init = init_again;
                    continue;
                }
            }
        }
    }
    pub fn wait<'a>(&'a self) -> &'a T {
        loop {
            match self.try_get() {
                Some(t) => return t,
                None => continue,
            }
        }
    }
    pub fn try_get<'a>(&'a self) -> Option<&'a T> {
        let state = self.state.load(Ordering::Acquire);

        if state != OnceState::Initialized as u8 {
            return None;
        }
        Some(unsafe { &*(self.value.get() as *const T) })
    }
    pub fn state(&self) -> OnceState {
        match self.state.load(Ordering::Acquire) {
            0 => OnceState::Uninitialized,
            1 => OnceState::Initializing,
            2 => OnceState::Initialized,
            _ => unreachable!(),
        }
    }
}
#[cfg(any(test, feature = "std"))]
impl<T: std::panic::UnwindSafe> std::panic::UnwindSafe for Once<T> {}

#[cfg(any(test, feature = "std"))]
impl<T: std::panic::RefUnwindSafe> std::panic::RefUnwindSafe for Once<T> {}

#[cfg(test)]
mod tests {
    use super::{
        Mutex, Once, OnceState, RawMutex, RawRwLock, RwLock, RwLockUpgradableReadGuard,
        RwLockWriteGuard,
    };

    use std::{sync::Arc, thread};

    #[test]
    fn singlethread_mutex() {
        let data = Mutex::new(2);
        assert_eq!(*data.lock(), 2);
        *data.lock() = 3;
        assert_eq!(*data.lock(), 3);
    }

    #[test]
    fn multithread_mutex() {
        let data = Arc::new(Mutex::new(2));
        let main_thread = thread::current();

        assert_eq!(*data.lock(), 2);

        {
            let data = Arc::clone(&data);
            thread::spawn(move || {
                *data.lock() = 3;
                main_thread.unpark();
            });
        }

        thread::park();
        assert_eq!(*data.lock(), 3);
    }
    #[test]
    fn multithread_rwlock() {
        // TODO: More complex test, or maybe this is done in an integration test.
        let data = Arc::new(RwLock::new(Vec::<u64>::new()));
        assert_eq!(&*data.read(), &[]);

        let threads = (0..4)
            .map(|index| {
                let data = Arc::clone(&data);
                thread::spawn(move || {
                    let mut write_guard = data.write();
                    write_guard.push(index);
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().unwrap();
        }
        let mut write_guard = data.write();
        write_guard.sort();

        let read_guard = RwLockWriteGuard::downgrade(write_guard);
        assert_eq!(&*read_guard, &[0, 1, 2, 3]);
    }

    #[test]
    fn singlethread_rwlock() {
        let data = RwLock::new(1);

        let intent_lock = data.upgradable_read();
        {
            let lock1 = data.read();
            let lock2 = data.read();
            let lock3 = data.read();

            assert_eq!(*lock1, 1);
            assert_eq!(*lock2, 1);
            assert_eq!(*lock3, 1);
            assert_eq!(*intent_lock, 1);
        }
        let mut write_lock = RwLockUpgradableReadGuard::upgrade(intent_lock);
        *write_lock = 2;

        let intent_lock_again = RwLockWriteGuard::downgrade_to_upgradable(write_lock);
        let lock1 = {
            let lock1 = data.read();
            let lock2 = data.read();

            assert_eq!(*intent_lock_again, 2);
            assert_eq!(*lock1, 2);
            assert_eq!(*lock2, 2);
            lock1
        };
        assert!(data.try_write().is_none());
        let lock3 = RwLockUpgradableReadGuard::downgrade(intent_lock_again);
        assert_eq!(*lock3, 2);
        assert_eq!(*lock1, 2);
    }

    #[test]
    fn intent_upgrade() {
        let data = RwLock::new(7);

        let upgradable = {
            let lock1 = data.try_read().unwrap();
            let lock2 = data.try_read().unwrap();
            let lock3 = data.try_read().unwrap();

            let upgradable = data.try_upgradable_read().unwrap();
            let upgrade_result = RwLockUpgradableReadGuard::try_upgrade(upgradable);
            assert!(
                upgrade_result.is_err(),
                "upgraded intent lock into exclusive lock while there were still readers"
            );

            assert_eq!(*lock1, 7);
            assert_eq!(*lock2, 7);
            assert_eq!(*lock3, 7);

            upgrade_result.err().unwrap()
        };
        let mut write_lock = RwLockUpgradableReadGuard::try_upgrade(upgradable).unwrap();
        *write_lock = 8;
        assert_eq!(*write_lock, 8);
    }

    #[test]
    fn singlethread_once() {
        let once = Once::<String>::uninitialized();
        assert_eq!(once.state(), OnceState::Uninitialized);
        assert_eq!(once.try_get(), None);
        once.initialize(String::from("Hello, world!"))
            .expect("once initialization failed");
        assert_eq!(once.state(), OnceState::Initialized);
        assert_eq!(once.try_get().map(String::as_str), Some("Hello, world!"));
        assert_eq!(once.wait(), "Hello, world!");
        assert!(once.initialize(String::from("Goodbye, world!")).is_err());
    }
    #[test]
    fn once_preinit() {
        let once = Once::<String>::initialized(String::from("Already initialized!"));
        assert_eq!(once.state(), OnceState::Initialized);
        assert_eq!(
            once.try_get().map(String::as_str),
            Some("Already initialized!")
        );
        assert_eq!(once.wait(), "Already initialized!");
    }
    #[test]
    fn once_with_panic_in_init() {
        let opinion = Arc::new(Once::<String>::new());
        let byte_str = b"Panicking is particul\xFFrly dangerous when dealing with unsafe!";

        let opinion_clone = Arc::clone(&opinion);

        // set panic hook to avoid messing up stdout
        std::panic::set_hook(Box::new(|_| {}));

        let join_handle = thread::Builder::new()
            .name(String::from("this thread should panic"))
            .spawn(move || {
                opinion_clone.call_once(|| String::from_utf8(byte_str.to_vec()).unwrap());
            })
            .unwrap();

        assert!(join_handle.join().is_err());
        assert_eq!(opinion.try_get(), None);
        assert_eq!(opinion.state(), OnceState::Initializing);
    }

    #[test]
    fn multithread_once() {
        let once = Arc::new(Once::new());
        assert_eq!(once.try_get(), None);
        assert_eq!(once.state(), OnceState::Uninitialized);

        let main_thread = thread::current();

        let values = [
            "initialized by first thread",
            "initialized by second thread",
            "initialized by third thread",
        ];

        let threads = values
            .iter()
            .copied()
            .map(|value| {
                let once = Arc::clone(&once);
                let main_thread = main_thread.clone();

                thread::spawn(move || {
                    once.call_once(|| value);
                    main_thread.unpark();
                })
            })
            .collect::<Vec<_>>();

        thread::park();
        assert!(once.initialize("initialized by main thread").is_err());
        assert!(once.try_get().is_some());
        assert!(values.contains(&once.wait()));

        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn const_init() {
        static mut _RWLOCK: RwLock<usize> = RwLock::const_new(RawRwLock::const_new(), 1);
        static mut _MUTEX: Mutex<usize> = Mutex::const_new(RawMutex::const_new(), 1);
    }

    // TODO: loom, although it doesn't seem to support const fn initialization
}
