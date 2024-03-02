#![feature(test)]
#![feature(stdsimd)]
#![feature(avx512_target_feature)]
#![allow(dead_code)]

use std::arch::x86_64::*;

extern crate test;

#[derive(Debug, Clone)]
pub struct SVecf32Owned {
    dims: u32,
    indexes: Vec<u32>,
    values: Vec<f32>,
}

impl SVecf32Owned {
    fn for_borrow(&self) -> SVecf32Borrowed<'_> {
        SVecf32Borrowed {
            dims: self.dims,
            indexes: &self.indexes,
            values: &self.values,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SVecf32Borrowed<'a> {
    dims: u32,
    indexes: &'a [u32],
    values: &'a [f32],
}

impl<'a> SVecf32Borrowed<'a> {
    #[inline(always)]
    pub fn indexes(&self) -> &[u32] {
        self.indexes
    }
    #[inline(always)]
    pub fn values(&self) -> &[f32] {
        self.values
    }
    #[inline(always)]
    pub fn len(&self) -> u16 {
        self.indexes.len().try_into().unwrap()
    }
    pub fn range(&self, start: usize, end: usize) -> SVecf32Borrowed<'a> {
        SVecf32Borrowed {
            dims: self.dims,
            indexes: &self.indexes[start..end],
            values: &self.values[start..end],
        }
    }
}

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn dot<'a>(lhs: SVecf32Borrowed<'a>, rhs: SVecf32Borrowed<'a>) -> f32 {
    let mut lhs_pos = 0;
    let mut rhs_pos = 0;
    let size1 = lhs.len() as usize;
    let size2 = rhs.len() as usize;
    let mut xy = 0.0;
    while lhs_pos < size1 && rhs_pos < size2 {
        let lhs_index = lhs.indexes()[lhs_pos];
        let rhs_index = rhs.indexes()[rhs_pos];
        if lhs_index == rhs_index {
            xy += lhs.values()[lhs_pos] * rhs.values()[rhs_pos];
            lhs_pos += 1;
            rhs_pos += 1;
        } else if lhs_index < rhs_index {
            lhs_pos += 1;
        } else {
            rhs_pos += 1;
        }
    }
    xy
}

#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn dot2<'a>(lhs: SVecf32Borrowed<'a>, rhs: SVecf32Borrowed<'a>) -> f32 {
    let mut lhs_pos = 0;
    let mut rhs_pos = 0;
    let size1 = lhs.len() as usize;
    let size2 = rhs.len() as usize;
    let mut xy = 0.;
    while lhs_pos < size1 && rhs_pos < size2 {
        let lhs_index = lhs.indexes()[lhs_pos];
        let rhs_index = rhs.indexes()[rhs_pos];
        let lhs_value = lhs.values()[lhs_pos];
        let rhs_value = rhs.values()[rhs_pos];
        xy += ((lhs_index == rhs_index) as u32 as f32) * lhs_value * rhs_value;
        lhs_pos += (lhs_index <= rhs_index) as usize;
        rhs_pos += (lhs_index >= rhs_index) as usize;
    }
    xy
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn _mm512_2intersect_epi32(a: __m512i, b: __m512i) -> (u16, u16) {
    use std::arch::asm;
    let res: u64;
    asm!(
        "vp2intersectd {0}, {1}, {2}",
        out(reg) res,
        in(zmm_reg) a,
        in(zmm_reg) b,
    );
    (res as u16, (res >> 16) as u16)
}

/// VP2INTERSECT emulation.
/// Díez-Cañas, G. (2021). Faster-Than-Native Alternatives for x86 VP2INTERSECT
/// Instructions. arXiv preprint arXiv:2112.06342.
#[inline(always)]
unsafe fn emulate_mm512_2intersect_epi32(a: __m512i, b: __m512i) -> (u16, u16) {
    let a1 = _mm512_alignr_epi32(a, a, 4);
    let b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    let m00 = _mm512_cmpeq_epi32_mask(a, b);
    let b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    let b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    let m01 = _mm512_cmpeq_epi32_mask(a, b1);
    let m02 = _mm512_cmpeq_epi32_mask(a, b2);
    let m03 = _mm512_cmpeq_epi32_mask(a, b3);
    let a2 = _mm512_alignr_epi32(a, a, 8);
    let m10 = _mm512_cmpeq_epi32_mask(a1, b);
    let m11 = _mm512_cmpeq_epi32_mask(a1, b1);
    let m12 = _mm512_cmpeq_epi32_mask(a1, b2);
    let m13 = _mm512_cmpeq_epi32_mask(a1, b3);
    let a3 = _mm512_alignr_epi32(a, a, 12);
    let m20 = _mm512_cmpeq_epi32_mask(a2, b);
    let m21 = _mm512_cmpeq_epi32_mask(a2, b1);
    let m22 = _mm512_cmpeq_epi32_mask(a2, b2);
    let m23 = _mm512_cmpeq_epi32_mask(a2, b3);
    let m30 = _mm512_cmpeq_epi32_mask(a3, b);
    let m31 = _mm512_cmpeq_epi32_mask(a3, b1);
    let m32 = _mm512_cmpeq_epi32_mask(a3, b2);
    let m33 = _mm512_cmpeq_epi32_mask(a3, b3);

    let m0 = m00 | m10 | m20 | m30;
    let m1 = m01 | m11 | m21 | m31;
    let m2 = m02 | m12 | m22 | m32;
    let m3 = m03 | m13 | m23 | m33;

    let res_a = m00
        | m01
        | m02
        | m03
        | (m10 | m11 | m12 | m13).rotate_left(4)
        | (m20 | m21 | m22 | m23).rotate_left(8)
        | (m30 | m31 | m32 | m33).rotate_right(4);

    let res_b = m0
        | ((0x7777 & m1) << 1)
        | ((m1 >> 3) & 0x1111)
        | ((0x3333 & m2) << 2)
        | ((m2 >> 2) & 0x3333)
        | ((0x1111 & m3) << 3)
        | ((m3 >> 1) & 0x7777);
    (res_a, res_b)
}

#[inline(always)]
fn emulate_mm512_2intersect_epi32_2(a: &[u32], b: &[u32]) -> (u16, u16) {
    let mut res1 = 0u16;
    let mut res2 = 0u16;
    for i in 0..16 {
        for j in 0..16 {
            let matchs = a[i] == b[j];
            res1 |= (matchs as u16) << i;
            res2 |= (matchs as u16) << j;
        }
    }
    (res1, res2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512cd")]
pub unsafe fn dot3<'a>(lhs: SVecf32Borrowed<'a>, rhs: SVecf32Borrowed<'a>) -> f32 {
    unsafe {
        const W: usize = 16;
        let mut lhs_pos = 0;
        let mut rhs_pos = 0;
        let size1 = lhs.len() as usize;
        let size2 = rhs.len() as usize;
        let lhs_size = size1 / W * W;
        let rhs_size = size2 / W * W;
        let mut xy = _mm512_setzero_ps();
        while lhs_pos < lhs_size && rhs_pos < rhs_size {
            let i_l = _mm512_loadu_epi32(lhs.indexes()[lhs_pos..].as_ptr().cast());
            let i_r = _mm512_loadu_epi32(rhs.indexes()[rhs_pos..].as_ptr().cast());
            let (m_l, m_r) = emulate_mm512_2intersect_epi32(i_l, i_r);
            let v_l = _mm512_loadu_ps(lhs.values()[lhs_pos..].as_ptr().cast());
            let v_r = _mm512_loadu_ps(rhs.values()[rhs_pos..].as_ptr().cast());
            let v_l = _mm512_maskz_compress_ps(m_l, v_l);
            let v_r = _mm512_maskz_compress_ps(m_r, v_r);
            xy = _mm512_fmadd_ps(v_l, v_r, xy);
            let l_max = lhs.indexes()[lhs_pos + W - 1];
            let r_max = rhs.indexes()[rhs_pos + W - 1];
            match l_max.cmp(&r_max) {
                std::cmp::Ordering::Less => {
                    lhs_pos += W;
                }
                std::cmp::Ordering::Greater => {
                    rhs_pos += W;
                }
                std::cmp::Ordering::Equal => {
                    lhs_pos += W;
                    rhs_pos += W;
                }
            }
        }
        dot(lhs.range(lhs_pos, size1), rhs.range(rhs_pos, size2)) + _mm512_reduce_add_ps(xy)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512cd")]
pub unsafe fn dot4<'a>(lhs: SVecf32Borrowed<'a>, rhs: SVecf32Borrowed<'a>) -> f32 {
    unsafe {
        const W: usize = 16;
        let mut lhs_pos = 0;
        let mut rhs_pos = 0;
        let size1 = lhs.len() as usize;
        let size2 = rhs.len() as usize;
        let lhs_size = size1 / W * W;
        let rhs_size = size2 / W * W;
        let mut xy = _mm512_setzero_ps();
        while lhs_pos < lhs_size && rhs_pos < rhs_size {
            let i_l = _mm512_loadu_epi32(lhs.indexes()[lhs_pos..].as_ptr().cast());
            let i_r = _mm512_loadu_epi32(rhs.indexes()[rhs_pos..].as_ptr().cast());
            let (m_l, m_r) = emulate_mm512_2intersect_epi32(i_l, i_r);
            let v_l = _mm512_loadu_ps(lhs.values()[lhs_pos..].as_ptr().cast());
            let v_r = _mm512_loadu_ps(rhs.values()[rhs_pos..].as_ptr().cast());
            let v_l = _mm512_maskz_compress_ps(m_l, v_l);
            let v_r = _mm512_maskz_compress_ps(m_r, v_r);
            xy = _mm512_fmadd_ps(v_l, v_r, xy);
            let l_max = lhs.indexes()[lhs_pos + W - 1];
            let r_max = rhs.indexes()[rhs_pos + W - 1];
            lhs_pos += (l_max <= r_max) as usize * W;
            rhs_pos += (l_max >= r_max) as usize * W;
        }
        dot(lhs.range(lhs_pos, size1), rhs.range(rhs_pos, size2)) + _mm512_reduce_add_ps(xy)
    }
}

pub fn random_svector(len: usize) -> SVecf32Owned {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut indexes: Vec<u32> = (0..len).map(|_| rng.gen_range(0..30000)).collect();
    indexes.sort_unstable();
    indexes.dedup();
    let values: Vec<f32> = (0..indexes.len())
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    SVecf32Owned {
        dims: 30000,
        indexes,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_dot() {
        const EP: f32 = 1e-6;
        let x = random_svector(300);
        let y = random_svector(350);
        dbg!(x.for_borrow(), y.for_borrow());
        let dot1 = dot(x.for_borrow(), y.for_borrow());
        let dot2 = dot2(x.for_borrow(), y.for_borrow());
        let dot3 = unsafe { dot3(x.for_borrow(), y.for_borrow()) };
        let dot4 = unsafe { dot4(x.for_borrow(), y.for_borrow()) };
        assert!(dot1 - dot2 < EP, "dot1: {}, dot2: {}", dot1, dot2);
        assert!(dot1 - dot3 < EP, "dot1: {}, dot3: {}", dot1, dot3);
        assert!(dot1 - dot4 < EP, "dot1: {}, dot4: {}", dot1, dot4);
    }

    #[test]
    fn test_mm512_2intersect_epi32() {
        let a = unsafe { _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) };
        let b =
            unsafe { _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31) };
        let (res_a, res_b) = unsafe { emulate_mm512_2intersect_epi32(a, b) };
        let (res_a2, res_b2) = emulate_mm512_2intersect_epi32_2(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        );
        assert_eq!(res_a, res_a2);
        assert_eq!(res_b, res_b2);
    }

    #[bench]
    fn bench_dot(b: &mut Bencher) {
        let x = random_svector(300);
        let y = random_svector(350);
        b.iter(|| dot(x.for_borrow(), y.for_borrow()));
    }

    #[bench]
    fn bench_dot_branchless(b: &mut Bencher) {
        let x = random_svector(300);
        let y = random_svector(350);
        b.iter(|| dot2(x.for_borrow(), y.for_borrow()));
    }

    #[bench]
    fn bench_dot_avx(b: &mut Bencher) {
        let x = random_svector(300);
        let y = random_svector(350);
        b.iter(|| unsafe { dot3(x.for_borrow(), y.for_borrow()) });
    }

    #[bench]
    fn bench_dot_avx_branchless(b: &mut Bencher) {
        let x = random_svector(300);
        let y = random_svector(350);
        b.iter(|| unsafe { dot4(x.for_borrow(), y.for_borrow()) });
    }
}
