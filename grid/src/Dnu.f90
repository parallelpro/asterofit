
module global_seismology

    use, intrinsic :: iso_fortran_env, only: real64
    implicit none

contains

    subroutine get_model_Dnu(mod_freq, mod_l, Dnu, numax, mod_n, mod_Dnu, mod_eps)
        real(real64), intent(in) :: mod_freq(:), Dnu, numax
        integer, intent(in) :: mod_l(:)
        integer, intent(in), optional :: mod_n(:)
        real(real64), intent(out) :: mod_Dnu, mod_eps
        real(real64) :: width, weight(:)
        integer :: l0(:), sidx(:), idx(:)
        real(real64) :: mod_freq_l0(:)
        integer :: n

        ! Calculate width based on parameters
        width = exp(0.9638 * log(numax) - 1.7145)

        ! Assign n
        l0 = pack([(i, i = 1, size(mod_l))], mod_l == 0)
        mod_freq_l0 = pack(mod_freq, mod_l == 0)
        sidx = sort_index(mod_freq_l0)
        mod_freq_l0 = mod_freq_l0(sidx)
        if (present(mod_n)) then
            n = pack(mod_n, mod_l == 0)
            n = n(sidx)
        else
            n = [(i, i = 1, size(mod_freq_l0))]
        endif

        ! Calculate weights
        weight = exp(-(mod_freq_l0-numax)**2./(2*width**2.))
        idx = weight > 1e-100
        if (count(idx) > 2) then
            ! Perform weighted linear fitting (need to implement or use a library)
            ! Calculate mod_Dnu and mod_eps from the fitting
            ! ...
        else
            mod_Dnu = real64'NaN'
            mod_eps = real64'NaN'
        endif
    end subroutine get_model_Dnu

end module model_dnu_mod