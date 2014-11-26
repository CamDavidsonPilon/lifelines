module estimation
  implicit none
contains
  function solve(A, b, n) result(x)
    ! solve the matrix equation A*x=b using LAPACK

    double precision, dimension(n,n), intent(in) :: A
    double precision, dimension(n), intent(in) :: b
    double precision, dimension(n) :: x

    integer :: pivot(n), ok

    integer, intent(in) :: n
    x = b

    ! find the solution using the LAPACK routine SGESV
    ! Same as np.linalg.solve calls
    call DGESV(n, 1, A, n, pivot, x, n, ok)
  end function

  subroutine cox_efron_values(x, beta, t, e, n, d, hessian, gradient)
    integer, intent(in) :: n, d
    ! Data
    double precision, dimension(d, n), intent(in) :: x
    ! Coefficients
    double precision, dimension(d), intent(in) :: beta
    ! Survival times
    double precision, dimension(n), intent(in) :: t
    ! Events 1 or 0
    logical, dimension(n), intent(in) :: e
    ! Results
    double precision, dimension(1, d), intent(out) :: gradient
    double precision, dimension(d, d), intent(out) :: hessian
    ! Locals
    integer :: tie_count, i, k
    double precision :: phi_i, risk_phi, tie_phi, c, denom
    double precision, dimension(1, d) :: phi_x_i, x_tie_sum, risk_phi_x, tie_phi_x, xi, partial_gradient, z
    double precision, dimension(d, d) :: phi_x_x_i, risk_phi_x_x, tie_phi_x_x, a1, a2

    ! Init to zero
    gradient(:, :) = 0
    hessian(:, :) = 0
    x_tie_sum(:, :) = 0
    risk_phi = 0
    tie_phi = 0
    risk_phi_x(:, :) = 0
    risk_phi_x_x(:, :) = 0
    tie_phi_x(:, :) = 0
    tie_phi_x_x(:, :) = 0

    ! init number of ties
    tie_count = 0

    ! Iterate backwards to utilize recursive relationship
    do i=n,1
      ! Preserve shape
      xi(1, :) = x(i, :)

      ! Calculate phi values
      phi_i = exp(dot_product(x(i, :), beta))
      phi_x_i = phi_i * xi
      phi_x_x_i = matmul(transpose(xi), xi) * phi_i

      ! Calculate sums of Risk set
      risk_phi = risk_phi + phi_i
      risk_phi_x = risk_phi_x + phi_x_i
      risk_phi_x_x = risk_phi_x_x + phi_x_x_i

      ! Calculate sums of Ties, if this is an event
      if (e(i)) then
        x_tie_sum = x_tie_sum + xi
        tie_phi = tie_phi + phi_i
        tie_phi_x = tie_phi_x + phi_x_i
        tie_phi_x_x = tie_phi_x_x + phi_x_x_i

        ! Keep track of count
        tie_count = tie_count + 1
      end if

      if (i > 1 .and. t(i-1) == t(i)) then
        ! There are more ties/members of the risk set
        cycle
      elseif (tie_count == 0) then
        ! Only censored with current time, move on
        cycle
      end if

      ! There was atleast one event and no more ties remain. Time to sum.
      partial_gradient(:, :) = 0

      do k=1,tie_count
        c = (k - 1.0) / tie_count

        denom = (risk_phi - c * tie_phi)
        z = (risk_phi_x - c * tie_phi_x)
        ! Gradient
        partial_gradient = partial_gradient + z / denom
        ! Hessian
        a1 = (risk_phi_x_x - c * tie_phi_x_x) / denom
        a2 = matmul(transpose(z), z) / (denom ** 2)

        hessian = hessian - (a1 - a2)
      end do

      ! Values outside tie sum
      gradient = gradient + x_tie_sum - partial_gradient

      !# reset tie values
      tie_count = 0
      x_tie_sum(:, :) = 0
      tie_phi = 0
      tie_phi_x(:, :) = 0
      tie_phi_x_x(:, :) = 0

    end do

  end subroutine


  function cox_newton_raphson(x, t, e, beta0, step, epsilon, n, d) result(beta)
    integer, intent(in) :: n, d
    ! Data
    double precision, dimension(n, d), intent(in) :: x
    ! Coefficients
    double precision, dimension(d), intent(in) :: beta0
    ! Survival times
    double precision, dimension(n), intent(in) :: t
    ! Events 1 or 0
    logical, dimension(n), intent(in) :: e
    ! Step stuff
    double precision, intent(in) :: step, epsilon
    ! Result and locals
    double precision, dimension(d) :: beta, delta
    double precision, dimension(1, d) :: gradient
    double precision, dimension(d, d) :: hessian

    ! Ignored later
    delta(:) = 1
    beta = beta0
    do while (sum(abs(delta)) < epsilon)
      call cox_efron_values(x, beta, t, e, n, d, hessian, gradient)
      delta = solve(-hessian, step * transpose(gradient), d)
      beta = beta + delta
      print *, delta
    end do

  end function

end module
