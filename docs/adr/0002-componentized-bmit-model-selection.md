# Componentized BMIT Model Selection

MOD_BMIT will compose explicit Balanced Motion and Internal Tide component model blocks (bm_model and it_model) rather than carrying flat, component-specific keys on the coupled model. MOD_BMIT owns the coupling choreography, total SSH, and cross-component diagnostics, while BM and IT dynamics own their own model options; this makes BM selectable between QG1L and QGSW (including QGSW qg or sw cores) and makes IT selection explicit with the current CSW1L implementation as the default. Legacy flat BMIT component keys are removed instead of mapped forward so new configurations state component ownership clearly.

**Consequences**

Existing MOD_BMIT experiment configs must move BM-specific and IT-specific options into bm_model and it_model. SW-core BM uses native prognostic U_BM, V_BM, and SSH_BM; Internal Tide Coupling always uses full BM velocity for advective terms, and the BM height used for equivalent-depth coupling is selected by bm_height_for_He.
