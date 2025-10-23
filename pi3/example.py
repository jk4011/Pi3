import torch
import argparse
from pi3.pi3_inference import pi3_inference, save_point_cloud

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--n_images", type=int, default=-1,
                        help="Number of images to sample from sequence. Default: -1 (all)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    
    # Determine precision
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Run inference with caching
    predictions = pi3_inference(
        data_path=args.data_path,
        interval=args.interval,
        n_images=args.n_images,
        ckpt=args.ckpt,
        precision=dtype
    )
    
    # Save point cloud
    save_point_cloud(predictions, args.save_path)
    
    print("Done.")

if __name__ == '__main__':
    main()