%% example2.m
%  The global example of nnstab: a stable plant that the controller only
%  nudges (small B and C). Then even the wide global sector of the network
%  is narrow enough, and the loop is exponentially stable for every
%  starting state x0 >= 0. Checked by simulation from large random starting
%  states.
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

[A, B, C, net] = nnstab.example_system('global');
res = nnstab.verify(A, B, C, net);
nnstab.report(res);

g  = res.global_check;
X0 = -100*log(rand(3, 100));                   % large random starting states
[t, X] = nnstab.simulate(A, B, C, net, X0, 10);
S = squeeze(sum(X, 2));                        % sum(x(t)), one column per solution
fprintf('\nSimulation of 100 solutions from large starting states:\n');
fprintf('  every state stays >= 0                       : %d\n', min(X(:)) >= -1e-12);
fprintf('  sum(x(t)) <= overshoot exp(-decay t) sum(x0) : %d\n', ...
        all(all(S <= g.overshoot*exp(-g.decay*t).*S(1, :)*(1 + 1e-9))));
fprintf('  largest sum(x0) and sum(x(10))               : %.4g, %.3g\n', max(S(1, :)), max(S(end, :)));
