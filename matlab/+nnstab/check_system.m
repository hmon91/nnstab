function [A, B, C, net] = check_system(A, B, C, net)
%CHECK_SYSTEM  Check the loop x' = A x + B pi(C x) and return A, B, C as matrices.
%   [A, B, C, net] = nnstab.check_system(A, B, C, net)
%
%   net is a network of nnsector.network. For one input, B may be given as
%   a row or a column; for one output, so may C. The method needs B >= 0,
%   C >= 0 and pi(0) = 0, so that x = 0 is an equilibrium; anything else is
%   an error.

    if ~isstruct(net) || ~all(isfield(net, {'W', 'b', 'activation', 'p', 'm'}))
        error('nnstab:system:net', 'net must be a network of nnsector.network.');
    end
    A = double(A);  B = double(B);  C = double(C);
    n = size(A, 1);
    if ~ismatrix(A) || size(A, 2) ~= n
        error('nnstab:system:size', 'A must be square, but it is %d x %d.', size(A, 1), size(A, 2));
    end
    if isvector(B) && size(B, 1) ~= n && numel(B) == n, B = B(:); end
    if isvector(C) && size(C, 2) ~= n && numel(C) == n, C = C(:).'; end
    if ~ismatrix(B) || size(B, 1) ~= n
        error('nnstab:system:size', 'B must have %d rows, like A.', n);
    end
    if ~ismatrix(C) || size(C, 2) ~= n
        error('nnstab:system:size', 'C must have %d columns, like A.', n);
    end
    if size(B, 2) ~= net.m
        error('nnstab:system:size', 'B has %d columns, but the network has %d outputs.', size(B, 2), net.m);
    end
    if size(C, 1) ~= net.p
        error('nnstab:system:size', 'C has %d rows, but the network has %d inputs.', size(C, 1), net.p);
    end
    if ~isreal(A) || ~isreal(B) || ~isreal(C) || any(~isfinite([A(:); B(:); C(:)]))
        error('nnstab:system:finite', 'A, B and C must be real and finite.');
    end
    if any(B(:) < 0) || any(C(:) < 0)
        error('nnstab:system:positive', ...
              'The method needs B >= 0 and C >= 0 (a positive input and output).');
    end
    pi0 = nnsector.forward(net, zeros(net.p, 1));
    if max(abs(pi0)) > 1e-10
        error('nnstab:system:equilibrium', ...
              ['The origin is not an equilibrium: pi(0) = %s. The method needs pi(0) = 0, ' ...
               'for example a network without biases with tanh, ReLU or leaky ReLU activations.'], ...
              mat2str(pi0.', 4));
    end
end
