# 1.  How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time? 

I actually processed the 50 files in 3 batches. First full run was actually just covering the first file in the dataset and it took ~455 seconds and based on that I estimated that whole 50 files will take approx. 6.4 hours but it was processed on one T4.The final version took around 3.3 hours but that is because it used 2 T4's so the time was nearly halved due to computing power not optimization or anything. And that 3.3 hours is the total of the three batches which took 2,484 , 4,913 , 4,622 , respectively.

# 2. Describe what your script does, step by step, from opening a Parquet file to writing an output line. 

The file is opened using PyArrow and then decoded into frames using PyAV. We use sampling , and keep 4 frames per second that will be analyzed.Those frames are given to the model, YOLO11l, which detects the frame for the 5 required objects.If one of those objects is present in half or more frames it's existence in that second is counted as true. And the number of people is median based, so the number of people in each frames is sorted and then median value is used. The results are then written as a line in jsonl which was later compressed into a gzip.

# 3. Which part of your final version takes the most time? How did you measure this?

According to calculations most of the time is going towards the model's processing. In a benchmark created by Claude where i tested six videos separately the decoding took 20 seconds while in total time taken was 72 seconds based on that i reached the conclusion that model and its related processes(like preprocessing and other overhead) was taking 72% of the total time consumed.
 
# 4. How did you check that your results stayed correct after you made the script faster? 

We first ran the YOLO11l on every frame of a few videos and then tested those same videos with sampling of 1,2,3,4,6,8 frames per second and then compared those sampled results with all frame results to see which was closest and also would be quick. At K=4 we achieved these results: 0.982 F1 score, 94.5% exact people counts, 99.6% within-one accuracy.

# 5. Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it? 

I did this assignment for learning purposes first and foremost so a lot of things were new to me so I let Claude take the lead. I assumed what it was doing was mostly correct. It gave a script to look for the downloaded files in the whole system so i told it that the CPU usage had become full , after which it corrected the script. It also suggested to send an email to Alethia AI to check how F1 is averaged but i rejected it because it was a weekend and we were not going to get a response.

# 6. Next steps. If you had one more week, what would you change, and why? 

After learning the things to some extent i realized that my script only handled the old rotate tag and not the new MP4 display matrix used by modern files so in an extra week i would have handled that. I would have sent the email asking how F1 is averaged to have a better understanding of things and would have helped with the validation.
