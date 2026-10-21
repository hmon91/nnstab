function [A, B, C, net] = example_system(kind)
%EXAMPLE_SYSTEM  A positive plant with nnstab.example_controller() in the loop.
%   [A, B, C, net] = nnstab.example_system('local')
%   [A, B, C, net] = nnstab.example_system('global')
%
%   'local' (default): a plant that is unstable on its own. The controller
%   makes the loop stable near the origin, where it pushes back hard, but
%   not far away, where its output levels off. The global test fails and a
%   local region is found.
%
%   'global': a stable plant that the controller only nudges. The global
%   test passes.

    if nargin < 1, kind = 'local'; end
    net = nnstab.example_controller();
    switch kind
        case 'local'
            A = [-3.0  3.5  2.0
                  2.0 -4.0  1.0
                  3.5  3.5 -5.0];
            B = [1.0; 0.5; 1.0];
            C = [1.0 1.0 0.5];
        case 'global'
            A = [-2.0  1.0  0.5
                  0.5 -1.5  0.5
                  1.0  0.5 -2.5];
            B = [0.1; 0.0; 0.2];
            C = [0.0 0.3 0.1];
        otherwise
            error('nnstab:example_system:kind', 'kind must be ''local'' or ''global''.');
    end
end
