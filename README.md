# spinning-rs
[![Build Status](https://travis-ci.org/4lDO2/spinning-rs.svg?branch=master)](https://travis-ci.org/4lDO2/spinning-rs)

A `#![no_std]` crate for spinlocks, intended to function similarly to [`spin`](https://crates.io/crates/spin), but with enhanced features such as SIX (shared-intent-exclusive) rwlocks, and [`lock_api`](https://crates.io/crates/lock_api).
