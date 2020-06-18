#![cfg_attr(not(test), no_std)]

use core::mem;
use core::sync::atomic::{self, AtomicBool, AtomicUsize, Ordering};

pub struct RawMutex {
    locked: AtomicBool,
}
unsafe impl lock_api::RawMutex for RawMutex {
    const INIT: Self = RawMutex { locked: AtomicBool::new(false) };

    type GuardMarker = lock_api::GuardSend;

    fn lock(&self) {
        while !self.try_lock() {
            atomic::spin_loop_hint();
        }
    }

    fn try_lock(&self) -> bool {
        self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }
    fn unlock(&self) {
        let prev = self.locked.fetch_and(false, Ordering::Release);
        debug_assert!(prev, "unlocking when not locked");
    }
}


pub struct RawRwLock {
    // The state of the rwlock is composed as follows:
    //
    // 1. The base bits, all the way up to ARCH_POINTER_WIDTH - 4, are used to count the number of
    //    current locks. These also include the one occasional intent or write lock, and the reason
    //    for this is to allow for more simple atomic operations, since x86 has no instrucion doing
    //    both bit tests and addition.
    // 2. RWLOCK_STATE_ACTIVE_WRITER_BIT indicates that the lock currently holds a writer. When
    //    acquiring a shared read lock, the counter may be incremented arbitrarily, but acquiring a
    //    read lock must always fail when this bit is set.
    // 3. RWLOCK_STATE_ACTIVE_INTENT_BIT denotes that the rwlock currently holds an intent lock.
    //    Attaining a read lock does ignores this bit, since intent locks only conflict with write
    //    locks, until they are upgraded to write locks.
    // 4. RWLOCK_STATE_PENDING_WRITER_BIT can be set at any time by a failed attempt at obtaining a
    //    write lock, due to read locks already being present. In order to prevent write lock
    //    starvation, new read locks cannot be acquired if this bit is set.
    state: AtomicUsize,
}

const RWLOCK_STATE_ACTIVE_WRITER_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 1);
const RWLOCK_STATE_ACTIVE_INTENT_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 2);

const RWLOCK_STATE_PENDING_WRITER_BIT: usize = 1 << (mem::size_of::<usize>() * 8 - 3);

const RWLOCK_STATE_EXTRA_MASK: usize = RWLOCK_STATE_ACTIVE_WRITER_BIT | RWLOCK_STATE_ACTIVE_INTENT_BIT | RWLOCK_STATE_PENDING_WRITER_BIT;
const RWLOCK_STATE_COUNT_MASK: usize = !RWLOCK_STATE_EXTRA_MASK;

impl RawRwLock {
    fn try_lock_exclusive_raw(&self) -> (bool, bool) {
        let prev_state = self.state.fetch_or(RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::Release);
        let was_previously_pending = prev_state & RWLOCK_STATE_PENDING_WRITER_BIT != 0;
        let success = self.state.compare_exchange(prev_state | RWLOCK_STATE_PENDING_WRITER_BIT, prev_state | RWLOCK_STATE_ACTIVE_WRITER_BIT, Ordering::Release, Ordering::Relaxed).is_ok();
        (success, was_previously_pending)
    }
}

unsafe impl lock_api::RawRwLock for RawRwLock {
    const INIT: Self = RawRwLock { state: AtomicUsize::new(0) };

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
            debug_assert_ne!(new_prev & !(RWLOCK_STATE_ACTIVE_WRITER_BIT | RWLOCK_STATE_ACTIVE_INTENT_BIT), 0, "overflow when subtracting rwlock counter");
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

        if !was_previously_pending {
            self.state.fetch_and(!RWLOCK_STATE_PENDING_WRITER_BIT, Ordering::Release);
        }

        success
    }

    // releases a read lock
    fn unlock_shared(&self) {
        let prev = self.state.fetch_sub(1, Ordering::Release);
        debug_assert_ne!(prev & RWLOCK_STATE_COUNT_MASK, 0, "corrupted state flags because of subtraction overflow, when release a shared lock");
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "releasing a shared lock while a write lock was held");
    }
    // releases a write lock
    fn unlock_exclusive(&self) {
        let prev = self.state.fetch_sub(RWLOCK_STATE_ACTIVE_WRITER_BIT | 1, Ordering::Release);
        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "corrupted state flags because a write lock release was tried when a write lock was not held");
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "releasing a write lock when an intent lock was held");
    }
}
unsafe impl lock_api::RawRwLockDowngrade for RawRwLock {
    // downgrades an exclusive lock to a shared lock
    fn downgrade(&self) {
        let prev = self.state.fetch_and(!RWLOCK_STATE_ACTIVE_WRITER_BIT, Ordering::Release);
        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "downgrading a write lock to a read lock when no write lock was held");
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "downgrading a write lock to a read lock when an intent lock was held");
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

        // begin by acquiring a read lock.
        if !self.try_lock_shared() { return false };

        let prev = self.state.fetch_or(RWLOCK_STATE_ACTIVE_INTENT_BIT, Ordering::Release);
        todo!()
    }
    // releases an intent lock
    fn unlock_upgradable(&self) {
        let prev = self.state.fetch_and(!RWLOCK_STATE_ACTIVE_INTENT_BIT, Ordering::Release);
    }
    // upgrades an intent lock into an exclusive lock
    fn upgrade(&self) {
    }

    // tries to upgrade an intent lock into an exclusive lock
    fn try_upgrade(&self) -> bool {
        let prev = self.state.fetch_xor(RWLOCK_STATE_ACTIVE_INTENT_BIT | RWLOCK_STATE_ACTIVE_WRITER_BIT, Ordering::Release);

        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "upgrading an intent lock into an exclusive lock when no intent lock was held");
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "upgrading an intent lock into an exclusive lock when an exclusive lock was held");

        prev & RWLOCK_STATE_COUNT_MASK == 1
    }
}
unsafe impl lock_api::RawRwLockUpgradeDowngrade for RawRwLock {
    // downgrades an exclusive lock to an intent lock
    fn downgrade_to_upgradable(&self) {
        let prev = self.state.fetch_xor(RWLOCK_STATE_ACTIVE_WRITER_BIT | RWLOCK_STATE_ACTIVE_INTENT_BIT, Ordering::Release);
        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "downgrading a write lock to an intent lock when no write lock was held");
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "downgrading a write lock to an intent lock when an intent lock was held");
    }
    // downgrades an intent lock into a shared lock
    fn downgrade_upgradable(&self) {
        let prev = self.state.fetch_and(!RWLOCK_STATE_ACTIVE_INTENT_BIT, Ordering::Release);
        debug_assert_eq!(prev & RWLOCK_STATE_ACTIVE_WRITER_BIT, 0, "downgrading an intent lock while a write lock was held");
        debug_assert_ne!(prev & RWLOCK_STATE_ACTIVE_INTENT_BIT, 0, "downgrading an intent lock where no intent lock was held");
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

#[cfg(test)]
mod tests {
    use super::{RwLock, Mutex};

    use std::{sync::Arc, thread};

    #[test]
    fn singlethread_mutex() {
        let data = Mutex::new(2);
        assert_eq!(*data.lock(), 2);
        *data.lock() = 3;
        assert_eq!(*data.lock(), 3);
    }

    #[test]
    fn multithread_mutex_loom() {
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
    fn singlethread_rwlock() {
        let data = RwLock::new(1);

        {
            let intent_lock = data.upgradable_read();

            let lock1 = data.read();
            let lock2 = data.read();
            let lock3 = data.read();
            
            assert_eq!(*lock1, 1);
            assert_eq!(*lock2, 1);
            assert_eq!(*lock3, 1);
            assert_eq!(*intent_lock, 1);
        }
    }
}
