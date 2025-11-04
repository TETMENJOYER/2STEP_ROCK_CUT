import pyvista as pv
import numpy as np
import PSO_for_cuts_pc as pso
import TS_for_cuts_pc as ts

def create_volumetric_slabs_z(block, slab_thickness, spacing):
    
    bounds = block.bounds
    z_min, z_max = bounds[4], bounds[5]
    
    slices = []
    current_z = z_min
    
    while current_z <= z_max:
        clip_bounds = [
            bounds[0], bounds[1],  
            bounds[2], bounds[3],  
            current_z, current_z + slab_thickness  
        ]
        
        sliced_block = block.clip_box(clip_bounds, invert=False)
        
        if sliced_block.n_points > 0:
            slices.append(sliced_block)
        
        current_z += slab_thickness + spacing
    
    return slices

def visualize_block_with_slabs_planes(block, slices):
    plotter = pv.Plotter()
    
    
    plotter.add_mesh(block, style='surface', color='lightblue', 
                    opacity=0.7, show_edges=True, label='Original Mesh')
    
    
    for i, slice_block in enumerate(slices):
        z_pos = slice_block.bounds[4]  
        
        plane = pv.Plane(center=[0, 0, z_pos], direction=(0, 0, 1), 
                        i_size=2, j_size=2)
        
        color = 'red' if i % 2 == 0 else 'yellow'
        plotter.add_mesh(plane, color=color, opacity=1 )
    
    plotter.add_legend()
    plotter.add_title("Исходный блок с плоскостями срезов")
    plotter.show()

def visualize_shifted_slabs(slabs):

    plotter = pv.Plotter()
    
    offset_distance = 0
    offset_step = 0.0  
    
    for i, slice_block in enumerate(slabs):
        shifted_slab = slice_block.copy()
        shifted_slab.points[:, 0] += offset_distance
        
        z_center = (slice_block.bounds[4] + slice_block.bounds[5]) / 2
        

        if hasattr(slabs[0], 'bounds'):
            all_bounds = [s.bounds for s in slabs]
            z_min = min(b[4] for b in all_bounds)
            z_max = max(b[5] for b in all_bounds)
            normalized_z = (z_center - z_min) / (z_max - z_min)
            
            
            import matplotlib.pyplot as plt
            color = plt.cm.viridis(normalized_z)
        else:
            color = 'blue'
        
        plotter.add_mesh(shifted_slab, color=color, opacity=1,
                        show_edges=False, line_width=2,
                        label=f'Z = {z_center}')
        
        offset_distance += offset_step
    
    plotter.add_legend()
    plotter.add_title("Смещенные срезы")
    plotter.show()

if __name__ == "__main__":

    block = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    slab_thickness = 0.3
    slice_spacing = 0.01
    
    slabs = create_volumetric_slabs_z(block, slab_thickness, slice_spacing)

    visualize_block_with_slabs_planes(block, slabs)
    
    visualize_shifted_slabs(slabs)