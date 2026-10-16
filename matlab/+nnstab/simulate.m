function [t, X] = simulate(A, B, C, net, x0, t_end, dt)
%SIMULATE  Simulate x' = A x + B pi(C x) with the classical Runge-Kutta method (RK4).
%   [t, X] = nnstab.simulate(A, B, C, net, x0, t_end)
%   [t, X] = nnstab.simulate(A, B, C, net, x0, t_end, dt)    (default dt = 0.01)
%
%   x0 is one starting state or an n x N matrix with one starting state per
%   column. t is a column of times. X(k, :, j) is the state at time t(k) of
%   the solution that starts at x0(:, j), so for one starting state X has
%   one row per time, like ode45.

    if nargin < 7, dt = 0.01; end
    [A, B, C, net] = nnstab.check_system(A, B, C, net);
    n = size(A, 1);
    if isvector(x0) && numel(x0) == n, x0 = x0(:); end
    if size(x0, 1) ~= n
        error('nnstab:simulate:size', 'x0 must have %d rows (one column per starting state).', n);
    end
    if ~(t_end > 0 && dt > 0)
        error('nnstab:simulate:time', 't_end and dt must be > 0.');
    end

    f = @(Z) A*Z + B*nnsector.forward(net, C*Z);
    steps = ceil(t_end/dt);
    h = t_end/steps;
    N = size(x0, 2);
    x = double(x0);
    X = zeros(steps + 1, n, N);
    X(1, :, :) = reshape(x, 1, n, N);
    for k = 1:steps
        k1 = f(x);
        k2 = f(x + 0.5*h*k1);
        k3 = f(x + 0.5*h*k2);
        k4 = f(x + h*k3);
        x = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
        X(k + 1, :, :) = reshape(x, 1, n, N);
    end
    t = linspace(0, t_end, steps + 1).';
end
