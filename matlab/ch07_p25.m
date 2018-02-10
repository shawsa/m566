clear all
close all




l = 7;

N = 2^l - 1;
tol = 1.e-6;  % should actually depend on N but never mind.


w = 10;
h = 1/(N+1);
n = N^2;
A = delsq(numgrid('S',N+2)) - diag(ones(n,1).*(w*h)^2);
A
%n = size(A,1);
b = A*ones(n,1);

% Solve using multigrid
t0 = cputime;
xmg = zeros(n,1); bb = norm(b);
flevel = log2(N+1);
for itermg = 1:100
    [xmg,rmg(itermg)] = poismg(A,b,xmg,flevel);
    if rmg(itermg)/bb < tol , break, end
end
time(1,1) = cputime - t0
itns(1,1) = itermg  

norm(A*xmg - b)