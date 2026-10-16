%% example1.m
%  The local example of nnstab: the global test fails, and a region of
%  attraction is found. The plant is unstable on its own. The controller is
%  a small tanh network without biases. Near y = 0 it pushes back hard,
%  which makes the loop stable; for large y its output levels off, and the
%  loop is unstable. The result is then checked by simulation from random
%  starting states in the region.
%
%  Run it from any folder. It needs the MATLAB version of nnsector: set the
%  environment variable NNSECTOR_MATLAB to its folder (the one that
%  contains +nnsector), or put the nnsector repository next to this one.
% =========================================================================
clear; clc; close all;

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root);                                  % the nnstab library folder
nns = getenv('NNSECTOR_MATLAB');
if isempty(nns), nns = fullfile(fileparts(fileparts(root)), 'nnsector', 'matlab'); end
addpath(nns);                                   % the nnsector library folder

[A, B, C, net] = nnstab.example_system('local');
fprintf('eigenvalues of A: %s\n\n', mat2str(real(eig(A)).', 3));

%% 1. The largest region
res = nnstab.verify(A, B, C, net);
nnstab.report(res);

%% 2. A smaller region, with decay rate at least 0.5
fast = nnstab.verify(A, B, C, net, 'MinDecay', 0.5);
fprintf('\n');
nnstab.report(fast);

%% 3. Check it by simulation from 200 random starting states in the region
c  = fast.local_check;
X0 = nnstab.sample_region(c, 200, 1);
[t, X] = nnstab.simulate(A, B, C, net, X0, 8);
V = squeeze(sum(X .* reshape(c.v, 1, []), 2));     % v'*x(t), one column per solution
fprintf('\nSimulation of 200 solutions that start in the region:\n');
fprintf('  every state stays >= 0           : %d\n', min(X(:)) >= -1e-12);
fprintf('  every solution stays in it       : %d\n', all(V(:) <= c.rho*(1 + 1e-9)));
fprintf('  v''*x(t) <= exp(-decay t) v''*x0  : %d\n', ...
        all(all(V <= V(1, :).*exp(-c.decay*t)*(1 + 1e-9) + 1e-12)));

% outside the region nothing is promised, and far away the loop is unstable
[~, Xfar] = nnstab.simulate(A, B, C, net, [4; 4; 4], 8);
fprintf('  from x0 = (4, 4, 4), outside it  : sum(x(8)) = %.3g\n', sum(Xfar(end, :)));

%% 4. Plot v'*x(t) / v'*x0 against the guarantee
figure;  hold on;  box on;  grid on;
h1 = plot(t, V ./ V(1, :), 'Color', [0.165 0.471 0.839], 'LineWidth', 0.5);
h2 = plot(t, exp(-c.decay*t), 'k--', 'LineWidth', 2);
set(gca, 'YScale', 'log');
xlabel('t');  ylabel('v''x(t) / v''x_0');
legend([h1(1), h2], {'200 solutions', sprintf('guarantee exp(-%.2f t)', c.decay)}, 'Location', 'southwest');
