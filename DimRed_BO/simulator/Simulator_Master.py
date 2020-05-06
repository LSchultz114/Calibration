from  subprocess import Popen
import os,shutil,sys
import numpy as np
import Simulation
import json_do

def Run_Task(DR_model,DR_input,directories,res_file,task):
   #directories: [home_dir, data_dir,sim_dir,py_dir,exp_dir]
   method=DR_model.method
   #file results in original subspace + all y outputs saved
   res_ofile=res_file.split(os.extsep)[0]+'_orig.'+res_file.split(os.extsep)[1]

   #####################################
   #Translate new sample to orig dim   #
   #####################################
   xhat=DR_model.Decode_X(DR_input)

   
   #####################################
   #Determine the y value and save it  #
   #####################################
   
   yhat=Simulation.Evaluate_y(xhat)[0][0]

   #####################################
   #Update everything                  #
   #####################################
   # Save the output into a results file in the appropriate folder
   DR_newline = str(yhat) + " " + ' '.join(map(str,DR_input)) + "\n"
   newline = str(yhat) + ' ' + ' '.join(map(str,xhat)) + "\n"
   DR_oldline = "P " + ' '.join(map(str,DR_input)) + "\n"
   
   #replace old with newly simulated
   with open(res_file,'r+') as outfile:
      temp = outfile.readlines()
   temp = [w.replace(DR_oldline,DR_newline) for w in temp]

   #print("replace %s with %s" % (oldline,newline))
   with open(res_file,'w+') as outfile:
      outfile.writelines(temp)

   #add newly sim to original subspace results file
   with open(res_ofile,'a+') as outfile:
      outfile.write(newline)
