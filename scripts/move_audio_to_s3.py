#----------------------------------------------
#   USAGE: 
#       This script will copy over local 
#       target dir to an AWS S3 instance
#
#   INPUTS:
#       1 --> relative path to local dir
#       2 --> path on s3 bucket     
# 
#   OUTPUTS:
#       None  
#----------------------------------------------




#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import boto
import time




#----------------------------------------------
#   main
#----------------------------------------------
def main():
    #get bucket name from argument list
    s3_bucket_name = sys.argv[2]
    #build up aws command (use recursive)
    aws_cli_command = "time aws s3 cp --recursive --quiet "
    #get local dir from argument list and convert to string
    local_dir = os.fsencode(sys.argv[1]).decode('UTF-8')
    #build path to s3 bucket
    aws_s3_bucket = " s3://" + s3_bucket_name + "/"

    #kick off transfer command and time it
    os.system(aws_cli_command + local_dir + aws_s3_bucket)




#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__":  
    #print quick message
    print("------------------")
    print("Copying Files To ---> {}".format(sys.argv[2]))
    print("------------------")  
    print("\n\n\n") 
    main()