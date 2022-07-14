rm -r "./results"
mkdir "./results"

python3 -m torch.distributed.launch \
--nproc_per_node=1 train.py --mode=test \
--world_size=1 --dataloaders=2 \
--test_input_poses_images=./poses/ \
--test_input_person_images=./character_sheet/ \
--test_output_dir=./results/ \
--test_checkpoint_dir=./weights/ 

echo Generating Video...
ffmpeg -r 30 -y -i ./results/%d.png -r 30 -c:v libx264 output.mp4 -r 30
echo DONE.
