@echo off


rd /S /Q  .\results
md .\results

FOR /F "delims=" %%I IN ("ffmpeg.exe") DO (if exist %%~$PATH:I (goto Hasffmpeg) else (goto NotHasffmpeg))


:NotHasffmpeg
echo.
echo Your windows env don' t have `ffmpeg`. Please install it!
echo.
goto END


:Hasffmpeg
python -m torch.distributed.launch ^
--nproc_per_node=1 train.py --mode=test ^
--world_size=1 --dataloaders=2 ^
--test_input_poses_images=./poses/ ^
--test_input_person_images=./character_sheet/ ^
--test_output_dir=./results/ ^
--test_checkpoint_dir=./weights/ 

echo Generating Video...

ffmpeg -r 30 ^
-y ^
-i "results\%%d.png" ^
-c:v libx264 ^
-pix_fmt yuv420p ^
output.mp4

echo DONE.

:END
pause