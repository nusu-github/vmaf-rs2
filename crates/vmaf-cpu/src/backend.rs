use crate::probe;

/// Target-neutral SIMD backend selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdBackend {
    /// Portable scalar fallback.
    #[default]
    Scalar,
    /// x86/x86_64 SSE2 baseline.
    X86Sse2,
    /// x86/x86_64 AVX2 + FMA backend.
    X86Avx2Fma,
    /// x86/x86_64 AVX-512 foundation backend (`avx512f` today).
    X86Avx512,
    /// AArch64 NEON/Advanced SIMD hook for future kernels.
    Aarch64Neon,
}

impl SimdBackend {
    /// Detects the best backend supported by the current process.
    pub fn detect() -> Self {
        probe::best_available_backend()
    }

    /// Detects the best backend supported by both the current process and the
    /// kernels implemented in this workspace.
    pub fn detect_effective() -> Self {
        Self::detect().effective()
    }

    /// Returns `true` when this backend is supported by the current process.
    pub fn is_available(self) -> bool {
        probe::backend_available(self)
    }

    /// Resolves a requested backend to the best kernel backend actually usable
    /// by this workspace on the current process.
    ///
    /// This keeps backend fallback policy in one place instead of duplicating
    /// AVX-512 → AVX2/SSE2/scalar handling across feature crates.
    pub fn effective(self) -> Self {
        if !self.is_available() {
            return Self::Scalar;
        }

        match self {
            Self::X86Avx512 => {
                if Self::X86Avx2Fma.is_available() {
                    Self::X86Avx2Fma
                } else if Self::X86Sse2.is_available() {
                    Self::X86Sse2
                } else {
                    Self::Scalar
                }
            }
            other => other,
        }
    }

    /// Returns a short, stable backend name for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::X86Sse2 => "x86-sse2",
            Self::X86Avx2Fma => "x86-avx2-fma",
            Self::X86Avx512 => "x86-avx512",
            Self::Aarch64Neon => "aarch64-neon",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SimdBackend;

    #[test]
    fn scalar_is_always_available() {
        assert!(SimdBackend::Scalar.is_available());
        assert_eq!(SimdBackend::Scalar.name(), "scalar");
    }

    #[test]
    fn detected_backend_is_available() {
        let detected = SimdBackend::detect();
        assert!(detected.is_available());
    }

    #[test]
    fn detected_effective_backend_is_available() {
        let detected = SimdBackend::detect_effective();
        assert!(detected.is_available());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_detection_never_reports_aarch64_backend() {
        let detected = SimdBackend::detect();

        assert!(!matches!(detected, SimdBackend::Aarch64Neon));
        assert!(!SimdBackend::Aarch64Neon.is_available());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_64_has_at_least_sse2() {
        assert_ne!(SimdBackend::detect(), SimdBackend::Scalar);
        assert!(SimdBackend::X86Sse2.is_available());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_avx512_request_falls_back_to_supported_kernel_backend() {
        let resolved = SimdBackend::X86Avx512.effective();

        if SimdBackend::X86Avx512.is_available() {
            assert!(matches!(
                resolved,
                SimdBackend::X86Avx2Fma | SimdBackend::X86Sse2
            ));
        } else {
            assert_eq!(resolved, SimdBackend::Scalar);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn unavailable_cross_arch_backend_falls_back_to_scalar() {
        assert_eq!(SimdBackend::Aarch64Neon.effective(), SimdBackend::Scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_detection_never_reports_x86_backend() {
        let detected = SimdBackend::detect();

        assert!(!matches!(
            detected,
            SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512
        ));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unavailable_cross_arch_backend_falls_back_to_scalar() {
        assert_eq!(SimdBackend::X86Sse2.effective(), SimdBackend::Scalar);
    }
}
