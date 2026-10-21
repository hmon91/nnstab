function w = largest_region_vector(M, c, rate)
%LARGEST_REGION_VECTOR  The least vector w with w >= c and w'*(M + rate*I) <= 0.
%   w = nnstab.largest_region_vector(M, c, rate)
%
%   M must be Metzler with M + rate*I Hurwitz, and c >= 0. Then
%   Z = -(M + rate*I)' is a nonsingular M-matrix, the vectors that meet both
%   conditions are closed under the entrywise minimum, and the least one
%   solves the linear complementarity problem
%       z = w - c >= 0,   s = Z*w >= 0,   z(i)*s(i) = 0.
%   It is found by Chandrasekaran's method in at most n solves. The least w
%   gives the largest region {x >= 0 : w'*x <= 1} among all such vectors:
%   it makes every entry of w as small as possible at once.

    c = c(:);
    n = numel(c);
    Z = -(M + rate*eye(n)).';
    scale = max(1, max(abs(Z(:))));
    r = Z*c;
    basis = false(n, 1);
    for it = 1:n+1
        z = zeros(n, 1);
        if any(basis)
            z(basis) = Z(basis, basis) \ (-r(basis));
        end
        w = c + z;
        s = Z*w;
        new = ~basis & (s < -1e-12*scale*max(1, max(abs(w))));
        if ~any(new)
            return
        end
        basis = basis | new;
    end
    error('nnstab:largest_region_vector:converge', ...
          'largest_region_vector did not converge; is M + rate*I Hurwitz?');
end
