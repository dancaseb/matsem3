import numpy as np

w_hat = np.array([0.1259391805448/2, 0.1259391805448/2, 0.1259391805448/2,0.1323941527885/2,0.1323941527885/2,0.1323941527885/2, 0.225/2])
A_hat = np.array([[0.1012865073235, 0.1012865073235], [0.7974269853531, 0.1012865073235], [0.1012865073235, 0.7974269853531], [0.4701420641051, 0.0597158717898],[0.4701420641051, 0.4701420641051],
[0.0597158717898, 0.4701420641051], [1/3, 1/3]])


# rows == number of quadrature points, cols == num of basis functions
def basic_func_P2(A):
    e0 = 1-A[:,0]-A[:,1]
    return np.column_stack([e0*(2*e0-1), A[:,0]*(2*A[:,0]-1), A[:,1]*(2*A[:,1]-1), 4*A[:,0]*A[:,1], 4*e0*A[:,1], 4*e0*A[:,0]])

def dbasic_func_de1(A):
    return np.column_stack([-3+4*A[:,0]+4*A[:,1], 4*A[:,0] - 1, np.zeros_like(A[:,0]), 4*A[:,1], -4*A[:,1],4 - 8*A[:,0]-4*A[:,1]])

def dbasic_func_de2(A):
    return np.column_stack([-3+4*A[:,0]+4*A[:,1], np.zeros_like(A[:,0]), 4*A[:,1] - 1, 4*A[:,0], 4-4*A[:,0]-8*A[:,1], -4*A[:,0]])


def compute_element_stiffness_P2(N, l, mu):
    # values of base functions in A_hat
    phi_A_hat = basic_func_P2(A_hat)
    # derivatives of base functions in A_hat
    dphi_de1 = dbasic_func_de1(A_hat)
    dphi_de2 = dbasic_func_de2(A_hat)

    K_T = np.zeros((2*N.shape[0], 2*N.shape[0]))

    C = np.array([[l + 2*mu, l, 0],
            [l, l + 2*mu, 0],
            [0, 0, mu]])
    phi_dx1 = np.zeros_like(dphi_de1)
    phi_dx2 = np.zeros_like(dphi_de2)
    
    for q in range(phi_A_hat.shape[0]):
        J11 = np.dot(dphi_de1[q, :], N[:, 0])
        J12 = np.dot(dphi_de1[q, :], N[:, 1])
        J21 = np.dot(dphi_de2[q, :], N[:, 0])
        J22 = np.dot(dphi_de2[q, :], N[:, 1])
        
        det_J = J11 * J22 - J12 * J21
        J_inv = np.array([[J22, -J12], [-J21, J11]]) / det_J
        

        # compute derivates of basis functions with respect to x_1 and x_2
        phi_dx1 = J_inv[0, 0] * dphi_de1[q, :] + J_inv[0, 1] * dphi_de2[q, :]
        phi_dx2 = J_inv[1, 0] * dphi_de1[q, :] + J_inv[1, 1] * dphi_de2[q, :]
    
        # construct B matrix for each quadrature point
        B_q = np.zeros((3, 2*N.shape[0]))
        for a in range(N.shape[0]):
            B_q[0, 2*a]   = phi_dx1[a]
            B_q[1, 2*a+1] = phi_dx2[a]
            B_q[2, 2*a]   = phi_dx2[a]
            B_q[2, 2*a+1] = phi_dx1[a]
        
        # adjusted weights
        w = w_hat[q] * np.abs(det_J)
        K_T += w * np.dot(B_q.T, np.dot(C, B_q))

    return K_T


if __name__ == "__main__":
    ############################################
    # User defined parameters
    # corner nodes of the element
    N_start = np.array([[0,0], [4,0], [1,2]])
    # material parameters
    l = 0.5 # lambda
    mu = 0.3
    ############################################

    N = np.zeros((6,2))
    N[0:3,:] = N_start
    # mid-side nodes
    for i in range(3):
        N[3+i,:] = 0.5*(N_start[(i+1)%3,:] + N_start[(i+2)%3,:])
    # N = np.array([[0,0], [3,0], [2,2], [2.5,1], [1,1], [1.5,0]])
    # N = np.array([[0,0], [1,0], [0,1], [0.5,0], [0.5,0.5], [0,0.5]])
    print("N:", N)
    K_T = compute_element_stiffness_P2(N, l, mu)
    print(np.linalg.det(K_T))
    print(K_T)