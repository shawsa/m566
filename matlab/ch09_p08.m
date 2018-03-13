N = 2^7-1;
n = N^2;
h = 1/(N+1);
a = .1;
itermax = 10^6;
%itermax = 0;
tol = 10^-6;


K = diag(repelem(4,N)) + diag(repelem(-1,N-1),1) + diag(repelem(-1,N-1),-1);
J = diag(repelem(-1,N-1),1) + diag(repelem(-1,N-1),-1);
A = kron(eye(N),K) + kron(J,eye(N));
A = sparse(A);

u = zeros(n,1);
for i=1:N
   for j=1:N
      x = h*j;
      y =                                                                                                                                                                                                                                                                                                                                                                                                                                    h*i;
      u((j-1)*N+i) = a*x*(1-x)*(1-y);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
   end
end

Jac = A - sparse(diag(exp(u)*h^2));
u_new = u + (Jac\(h^2*exp(u) - A*u));
for i=1:itermax
    if norm(u_new-u, inf)<tol
       break                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
       disp(i)
    end
    u = u_new;
    Jac = A - sparse(diag(exp(u)*h^2));
    u_new = u + (Jac\(h^2*exp(u) - A*u));
end

disp(i)
Z = reshape(u, [N,N]);
figure
mesh(Z)
