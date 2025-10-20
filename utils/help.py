import torch

# ============== 辅助函数 ==============

def get_reference_points(spatial_shapes, device):
    """
    生成参考点坐标
    
    Args:
        spatial_shapes: [num_levels, 2] - 每个特征层的 (H, W)
        device: 设备
    
    Returns:
        reference_points: [total_pixels, num_levels, 2] - 归一化的参考点
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        # 生成网格坐标
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)
        )
        # 归一化到 [0, 1]
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref = torch.stack((ref_x, ref_y), -1)  # [H*W, 2]
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, dim=0)  # [total_pixels, 2]
    # 扩展到所有层级 [total_pixels, num_levels, 2]
    reference_points = reference_points[:, None, :].repeat(1, len(spatial_shapes), 1)
    
    return reference_points


def get_level_start_index(spatial_shapes):
    """
    获取每个层级的起始索引
    
    Args:
        spatial_shapes: [num_levels, 2]
    
    Returns:
        level_start_index: [num_levels] - 每层起始索引
    """
    num_pixels_per_level = [H * W for H, W in spatial_shapes]
    level_start_index = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(torch.tensor(num_pixels_per_level[:-1]), dim=0)
    ])
    return level_start_index