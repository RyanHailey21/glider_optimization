import aerosandbox as asb
import aerosandbox.numpy as np
import casadi as ca

def build_surrogate(af_name):
    print(f"Generating aerodynamic grid for {af_name}...")
    af = asb.Airfoil(af_name)
    
    # Grid of operating points
    alphas = np.linspace(-10, 15, 26)  # 1 degree resolution
    Res    = np.array([20e3, 40e3, 60e3, 80e3, 100e3, 150e3])
    
    A, R = np.meshgrid(alphas, Res, indexing='ij')
    
    # Flatten for batch evaluation if possible, or loop
    CL = np.zeros_like(A)
    CD = np.zeros_like(A)
    Cm = np.zeros_like(A)
    
    # NuralFoil call
    try:
        from neuralfoil import get_aero_from_airfoil
        for i, a in enumerate(alphas):
            for j, re in enumerate(Res):
                res = get_aero_from_airfoil(af, alpha=a, Re=re)
                # NeuralFoil sometimes returns 1D arrays even for scalar inputs
                CL[i, j] = float(np.atleast_1d(res["CL"])[0])
                CD[i, j] = float(np.atleast_1d(res["CD"])[0])
                Cm[i, j] = float(np.atleast_1d(res["CM"])[0])
    except ImportError:
        print("NeuralFoil not installed natively, falling back to XFoil if possible")
        import sys; sys.exit(1)
        
    print("Grid built. Creating interpolants...")
    # CasADi expects 1D arrays for grids, and flattened 1D array for data (Fortran order for meshgrid 'ij')
    cl_spline = ca.interpolant("CL", "bspline", [alphas, Res], CL.ravel(order='F'))
    cd_spline = ca.interpolant("CD", "bspline", [alphas, Res], CD.ravel(order='F'))
    cm_spline = ca.interpolant("Cm", "bspline", [alphas, Res], Cm.ravel(order='F'))
    
    print("Evaluating interpolant at alpha=5.0, Re=60000:")
    test_pt = [5.0, 60000.0]
    print(f"CL: {cl_spline(test_pt)}")
    print(f"CD: {cd_spline(test_pt)}")
    print(f"Cm: {cm_spline(test_pt)}")
    
if __name__ == "__main__":
    build_surrogate("sd7037")
