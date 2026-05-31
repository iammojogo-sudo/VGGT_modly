# VGGT Scene Reconstruction

Multi-view scene and room reconstruction using Meta's VGGT-1B model. Feed it a folder of overlapping photos of a room and it reconstructs the observed geometry as a colored mesh.

## Setup

Install via Modly's extension manager. The setup script downloads VGGT weights (~2.8 GB) and installs dependencies including PyTorch (CUDA-matched to your GPU), Open3D (optional, for Poisson meshing), and the VGGT model repository.

## Usage

Point the `Images Folder` parameter at a directory containing your photos. Every image in that folder is used. If you don't specify a folder, only the single image wired into the node is processed (not recommended for rooms — you need several overlapping views).

### Key Parameters

**Max Views** — How many photos to feed to VGGT. Scales with memory; lower on weaker GPUs.

**Confidence Filter** — Drops the lowest-confidence depth estimates before meshing. Higher values (70–85%) remove more noise but thin the cloud; 0 keeps everything.

**Output** — Mesh (Poisson reconstruction) or raw point cloud.

**Downsample** — Voxel size for thinning the cloud. Larger = faster, smoother; 0 disables it.

**Mesh Detail** — Poisson octree depth. Higher adds detail but more memory and spurious geometry.

**Planar Cleanup** — Fit dominant flat planes (walls, floor, ceiling) to the reconstruction. Off keeps the raw mesh. RANSAC flattens each plane. Manhattan snap also snaps them to orthogonal axes for true 90-degree corners.

## Output

A GLB mesh with vertex colors. If meshing fails or is disabled, exports a point cloud instead.

## What It Works On

- Rooms, indoor scenes with overlapping photo coverage.
- Photos taken while walking through the space (not spinning in one spot).
- Consistent lighting and white balance across shots.

## What It Doesn't Handle

- Single objects (use Hunyuan3D-2mv for that).
- Photos taken from wildly different positions with no overlap.
- Heavily backlit or silhouetted areas.

## Notes

VGGT assumes each input image covers some part of the scene and tries to align all of them into one coordinate frame. If photos don't overlap or are taken from very different scales, alignment can fail — you'll see the output "fold" or "petal" in strange ways. The Planar Cleanup modes can help abstract that into a clean boxy shell, but they won't fix the fundamental misalignment.

For best results: move slowly between shots, maintain good overlap, keep the camera at a consistent orientation (don't mix portrait and landscape), and capture several angles of the space.

## Open3D

Poisson meshing requires Open3D, which may not install on some Python versions (particularly 3.13+). If it fails, use Point Cloud output or Planar Cleanup modes instead — both are Open3D-free.
