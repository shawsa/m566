clear all
close all
% This code is a modified version of the code used in example 7.7
% https://www.mathworks.com/support/books/book69732.html

l = 7;
ns = zeros(4,1);
iterations = zeros(4,1);
for i=1:4
    l = i+3;
    N = 2^l - 1;
    
    %N = 2^7 -1;
    tol = 1e-6;  % should actually depend on N but never mind.


    w = 10;
    h = 1/(N+1);
    n = N^2;
    ns(i,1) = n;
    A = delsq(numgrid('S',N+2)) - diag(ones(n,1).*(w*h)^2);
    %n = size(A,1);
    b = ones(n,1)*h^2;

    % Solve using multigrid
    xmg = zeros(n,1); bb = norm(b);
    flevel = log2(N+1);
    for itermg = 1:20
       [xmg,res] = poismg(A,b,xmg,flevel,tol);
       if res/bb < tol
              break;
       end
    end
    %[xmg, ~, ~, itermg] = minres(A,b,tol,1000);
    iterations(i,1) = itermg;
    fprintf("%i: %e\n", l, norm(A*xmg - b));
end

hold on
plot(ns, iterations,'o');
xlabel("n");
ylabel("Mutigrid Iterations");
ylim([0,10]);
hold off

%[~,pd] = chol(A);
%pd
%%

N = 2^7-1;
w = 1;
h = 1/(N+1);
n = N^2;
A = delsq(numgrid('S',N+2)) - diag(ones(n,1).*(w*h)^2);
b = ones(n,1)*h^2;

[x, ~, ~, iter] = minres(A,b,1e-6,2000);
norm(A*x-b)
iter

