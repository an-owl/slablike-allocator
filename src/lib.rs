#![cfg_attr(not(test),no_std)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]

mod slab_meta;

use core::ptr::NonNull;
use slab_meta::{
    SlabMetadata,
    meta_bitmap_size
};

pub struct SlabLike<A: core::alloc::Allocator, T, const SLAB_SIZE: usize> where [(); meta_bitmap_size::<T, SLAB_SIZE>()]: {
    alloc: A,
    head: Option<NonNull<Slab<T, SLAB_SIZE>>>,
}

#[repr(C)]
struct Slab<T, const SLAB_SIZE: usize> where [(); meta_bitmap_size::<T, SLAB_SIZE>()]: {
    slab_metadata: SlabMetadata<T, SLAB_SIZE>,
}
