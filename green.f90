module green_function
    contains
    subroutine green(x, y, z, n, xs, ys, zs, omg, ulen, Gf, dGdx, dGdy, dGdz)
        implicit none
        real, intent(in) :: xs, ys, zs, omg, ulen
        integer, intent(in) :: n
        real, dimension(n), intent(in) :: x, y, z
        complex, dimension(n), intent(out) :: Gf, dGdx, dGdy, dGdz
        real, dimension(:), allocatable :: rho, d, h, v, R0, R1
        complex, dimension(:), allocatable :: dGdrho
        real :: f, pi, R0_t, R1_t, tol
        real, dimension(:), allocatable :: J0, J1
        integer :: i 
        complex :: j 
        real :: g = 9.80665

        allocate(rho(n)); allocate(J0(n)); allocate(J1(n)); allocate(d(n)); allocate(h(n));
        allocate(v(n)); allocate(R0(n)); allocate(R1(n)); allocate(dGdrho(n))

        pi = 4.0 * atan(1.0)
        f = ((omg**2)*ulen)/g
        tol = 1.0e-7
        j = (0,1)

        rho = sqrt((x-xs)**2 + (y-ys)**2)
        h = f * rho
        v = f * (z + zs)
        d = sqrt(h ** 2 + v ** 2)

        ! Bessel Function
        call bessel_J0(n, h, J0)
        call bessel_J1(n, h, J1)

        !Gradif function
        do i = 1, n
            call GRADIF(h(i), v(i), R0_t, R1_t)
            R0(i) = R0_t
            R1(i) = R1_t
        end do

        ! domain of h and v
        where((h .lt. 0.00) .or. (v .gt. 0.00)) R0 = 0.0
        where((h .lt. 0.00) .or. (v .gt. 0.00)) R1 = 0.0
        where(isnan(R0)) R0 = 0.0
        where(isnan(R1)) R1 = 0.0

        Gf = 2.0 * f *(R0 - j* pi* J0 * exp(v)) / ulen
        dGdrho = -2.0 * (f**2) * (R1 - j * pi *J1* exp(v))
        dGdx = - (dGdrho) * (x-xs) / rho / ulen ** 2
        dGdy = - (dGdrho) * (y-ys) / rho / ulen ** 2
        dGdz = -2.0 * f**2 * (1.0/d + R0 - j* pi* J0* exp(v)) / ulen**2

        !less than tol -> 0
        where(rho < tol) dGdx = 0.0
        where(rho < tol) dGdy = 0.0
        where(d < tol) dGdz = 0.0
        return
        deallocate(rho, h, v, J0, J1, d, R0, R1, dGdrho)
    end subroutine green

    subroutine bessel_J0(n, h, J0)
        integer, intent(in) :: n
        real, dimension(n), intent(in) :: h
        real, dimension(n), intent(out) :: J0
        real, dimension(n) :: y, ax, z, xx
        real p1,p2,p3,p4,p5,q1,q2,q3,q4,q5,r1,r2,r3,r4, &
            r5,r6,s1,s2,s3,s4,s5,s6
        SAVE p1,p2,p3,p4,p5,q1,q2,q3,q4,q5,r1,r2,r3,r4,r5,r6, &
            s1,s2,s3,s4,s5,s6
        DATA p1,p2,p3,p4,p5/1.e0,-.1098628627e-2,.2734510407e-4, &
            -.2073370639e-5,.2093887211e-6/, q1,q2,q3,q4,q5/-.1562499995e-1, &
            .1430488765e-3,-.6911147651e-5,.7621095161e-6,-.934945152e-7/ 
        DATA r1,r2,r3,r4,r5,r6/57568490574.e0,-13362590354.e0,651619640.7e0,&
            -11214424.18e0,77392.33017e0,-184.9052456e0/,&
            s1,s2,s3,s4,s5,s6/57568490411.e0,1029532985.e0,&
            9494680.718e0,59272.64853e0,267.8532712e0,1.e0/
        where(abs(h).lt.8.)
            y=h**2
            J0 = (r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6))))) &
                /(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6)))))
        elsewhere
            ax = abs(h)
            z=8.0/ax
            y= z**2
            xx=ax - .785398164
            J0 = sqrt(.636619772/ax)*(cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y &
                 *p5))))-z*sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5)))))
        end where
        return

    end subroutine bessel_J0

    subroutine bessel_J1(n, h, J1)
        integer, intent(in) :: n
        real, dimension(n), intent(in) :: h
        real, dimension(n) :: y, ax, z, xx
        real, dimension(n), intent(out) :: J1
        real p1,p2,p3,p4,p5,q1,q2,q3,q4,q5,r1,r2,r3,r4, &
            r5,r6,s1,s2,s3,s4,s5,s6
        SAVE p1,p2,p3,p4,p5,q1,q2,q3,q4,q5,r1,r2,r3,r4,r5,r6, &
            s1,s2,s3,s4,s5,s6
        DATA r1,r2,r3,r4,r5,r6/72362614232.e0,-7895059235.e0,242396853.1e0, &
            -2972611.439e0,15704.48260e0,-30.16036606e0/, &
            s1,s2,s3,s4,s5,s6/144725228442.e0,2300535178.e0, &
            18583304.74e0,99447.43394e0,376.9991397e0,1.e0/
        DATA p1,p2,p3,p4,p5/1.e0,.183105e-2,-.3516396496e-4,.2457520174e-5, &
            -.240337019e-6/, q1,q2,q3,q4,q5/.04687499995e0,-.2002690873e-3, &
            .8449199096e-5,-.88228987e-6,.105787412e-6/

        
        where(abs(h).lt.8.)
            y=h**2
            J1=h*(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6))))) &
                /(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6)))))
        elsewhere
            ax=abs(h)
            z=8.0/ax
            y=z**2
            xx=ax-2.356194491
            J1 = sqrt(.636619772/ax)*(cos(xx)*(p1+y*(p2+y*(p3+y*(p4+y &
                *p5))))-z*sin(xx)*(q1+y*(q2+y*(q3+y*(q4+y*q5))))) &
                *sign(1.,h)
        end where
        return
    end subroutine bessel_J1
end module green_function
