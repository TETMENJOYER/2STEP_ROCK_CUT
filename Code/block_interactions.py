import pyvista as pv
import numpy as np

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
                    opacity=1, show_edges=True)
    
    
    for i, slice_block in enumerate(slices):
        z_pos = slice_block.bounds[4]  
        
        plane = pv.Plane(center=[0, 0, z_pos], direction=(0, 0, 1), 
                        i_size=2, j_size=2)
        
        color = 'red' if i % 2 == 0 else 'yellow'
        plotter.add_mesh(plane, color=color, opacity=1 )
    
    plotter.add_legend()
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
                        show_edges=False, line_width=2)
        
        offset_distance += offset_step
    
    plotter.show()

    
import pyvista as pv
import numpy as np

def slice_polygon_by_height(block, slice_height, offset=0.0, exclude_empty=True):
    
    # Получаем границы полигона по Z
    bounds = block.bounds
    z_min, z_max = bounds[4], bounds[5]
    total_height = z_max - z_min
    
    # Рассчитываем позиции срезов
    start_z = z_min + offset
    end_z = z_max
    
    # Создаем массив позиций срезов с заданным шагом
    slice_positions = np.arange(start_z, end_z, slice_height)
    
    # Добавляем последний срез, если он близко к верхней границе
    if len(slice_positions) == 0 or slice_positions[-1] < end_z - slice_height * 0.1:
        slice_positions = np.append(slice_positions, end_z)
    
    
    # Создаем сечения
    slices = []
    valid_positions = []
    
    for z_pos in slice_positions:
        x_center = (bounds[0] + bounds[1]) / 2
        y_center = (bounds[2] + bounds[3]) / 2
        
        origin = [x_center, y_center, z_pos]
        normal = [0, 0, 1]
        
        try:
            slice_result = block.slice(normal=normal, origin=origin)
            
            if exclude_empty:
                if slice_result.n_points >= 3: 
                    slices.append(slice_result)
                    valid_positions.append(z_pos)
            else:
                slices.append(slice_result)
                valid_positions.append(z_pos)
                
        except Exception as e:
            print(f"Предупреждение: ошибка при создании среза Z={z_pos:.3f}: {e}")
    
    
    return slices

if __name__ == "__main__":

    block = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    slab_thickness = 0.3
    slice_spacing = 0.01
    
    slabs = create_volumetric_slabs_z(block, slab_thickness, slice_spacing)

    visualize_block_with_slabs_planes(block, slabs)
    
    visualize_shifted_slabs(slabs)