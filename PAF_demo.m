%% Implementation of the Perturbed Amplitude Flow algorithm proposed in the paper
%  `` Perturbed amplitude flow for phase retrieval by Gao, B., et al.(2020).``
%  The code below is adapted from implementation of the Wirtinger Flow
% algorithm implemented by E. Candes, X. Li, and M. Soltanolkotabi.


clear
close all

%% Set Parameters
if exist('Params')                == 0,  Params.n2          = 1;    end
if isfield(Params, 'n1')          == 0,  Params.n1          = 1024; end             % signal dimension
if isfield(Params, 'm')           == 0,  Params.m           = 5 * Params.n1;  end     % number of measurements
if isfield(Params, 'cplx_flag')   == 0,  Params.cplx_flag   = 1;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'T')           == 0,  Params.T           = 150;  end    	% number of iterations
if isfield(Params, 'mu')          == 0,  Params.mu          = 0.2;  end		% step size / learning parameter
if isfield(Params, 'npower_iter') == 0,  Params.npower_iter = 30;   end		% number of power iterations

n           = Params.n1;    
m           = Params.m;         
cplx_flag	= Params.cplx_flag;  % real-valued: cplx_flag = 0;  complex-valued: cplx_flag = 1;    
mu          = 2.5;  % step size 
gamma       = 1/sqrt(3)*(1-cplx_flag) + 1/2*cplx_flag;
alp         = 2*(1-cplx_flag) + 1*cplx_flag; % perturbation coefficient
npower_iter = Params.npower_iter;           % number of power iterations 

%% make the data
x = randn(n,1)+ cplx_flag * 1i * randn(n,1);

Amatrix = (randn(m,n)+ cplx_flag * 1i * randn(m,n)) / (sqrt(2)^cplx_flag);

 
A  = @(I) Amatrix  * I;
At = @(Y) Amatrix' * Y;
y  = abs(A(x)); % y_i=|a_i x|

tic
%% Initialization
z0 = randn(n,Params.n2); z0 = z0/norm(z0,'fro');    % Initial guess 

lamd2 = sum(y(:).^2)/numel(y(:));
ytrf = gamma - exp(-y.^2 / lamd2); % transformed version of amplitude measurement 
for tt = 1:npower_iter                    
    z0 = At( ytrf.* (A(z0)) ); 
    z0 = z0 / norm(z0,'fro');
end
z0 = sqrt(lamd2) * z0;             % Apply scaling


%% Gradient-descent iteration
Relerrs=zeros(Params.T+1,1);
z = z0;
Relerrs(1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

eta = sqrt(alp) * y; % perturbation
t=1;
while t<=Params.T
    yz = Amatrix * z;
    grad = 1/m * At(yz - sqrt(y.^2+eta.^2).*yz./sqrt(abs(yz).^2 + eta.^2));
    z = z - mu  *  grad;
    Relerrs(t+1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
    t=t+1;
end
toc

figure, plot(1:n, real(x), 'k-', 1:n, real(exp(-1i*angle(trace(x'*z)))*z), 'r-', 'linewidth', 2);
legend('original', 'retrieval')

T = Params.T;
fprintf('Relative error before initialization: %f\n', Relerrs(1))
fprintf('PAF Relative error after %d iterations: %14f\n', t, Relerrs(t))
 
figure, h1=semilogy(0:T,Relerrs,'-r'); 
xlabel('Iteration'), ylabel('Relative error (log10)'),title('Relative error vs. iteration count')