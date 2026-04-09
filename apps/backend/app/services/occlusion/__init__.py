"""
ML-native dental occlusion analysis package.

Modules:
- arch_encoder: DGCNN backbone for per-tooth point cloud encoding
- occlusal_losses: Differentiable loss functions (Chamfer, overlap, uniformity, etc.)
- se3_transforms: SE(3) rigid body transforms via pytorch3d
- collision_detection: Differentiable + BVH collision detection
- landmark_detector: Learned dental landmark extraction
- occlusion_service: Main ML service (OcclusionService)
"""
