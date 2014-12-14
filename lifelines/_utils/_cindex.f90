! Calculates the concordance index (C-index) between two series
! of event times. The first is the real survival times from
! the experimental data, and the other is the predicted survival
! times from a model of some kind.
!
! The concordance index is a value between 0 and 1 where,
! 0.5 is the expected result from random predictions,
! 1.0 is perfect concordance and,
! 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)
!
! Parameters:
!   event_times: a (nx1) array of observed survival times.
!   predicted_event_times: a (nx1) array of predicted survival times.
!   event_observed: a (nx1) array of censorship flags, 1 if observed,
!                   0 if not. Default assumes all observed.
function concordance_index (event_times, predictions, event_observed, rows) result(cindex)
  double precision :: cindex, paircount, csum
  integer, intent(in) :: rows
  double precision, dimension(rows), intent(in) :: event_times, predictions, event_observed
  double precision :: time_a, pred_a, event_a, time_b, pred_b, event_b
  integer :: a, b
  logical :: valid_pair

  ! Default values
  paircount = 0
  csum = 0
  cindex = 0

  do a=1, rows-1
    time_a = event_times(a)
    pred_a = predictions(a)
    event_a = event_observed(a)
    ! Start at a+1 to avoid double counting
    do b=a+1, rows
      time_b = event_times(b)
      pred_b = predictions(b)
      event_b = event_observed(b)

      ! Check if it's a valid comparison
      if (event_a == 1 .and. event_b == 1) then
        ! Two events can always be compared
        valid_pair = .true.
      else if (event_a == 1 .and. time_a < time_b) then
        ! If b is censored, then a must have event first
        valid_pair = .true.
      else if (event_b == 1 .and. time_b < time_a) then
        ! If a is censored, then b must have event first
        valid_pair = .true.
      else
        ! Not valid to compare this pair
        valid_pair = .false.
      end if

      if (valid_pair) then
        paircount = paircount + 1

        ! Check concordance
        if (pred_a == pred_b) then
          csum = csum + 0.5
        else if (time_a < time_b .and. pred_a < pred_b) then
          csum = csum + 1.0
        else if (time_b < time_a .and. pred_b < pred_a) then
          csum =csum + 1.0
        !else
        ! csum = csum + 0
        end if
      end if
    end do
  end do

  ! Calculate c-index.
  if (paircount > 0) then
    cindex = csum / paircount
  else
    cindex = 0.5
  end if

end function concordance_index
