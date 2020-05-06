import json,os
import numpy as np

##reads necessary json files

 
def Stats_2L(data_dir):
    con_file= os.path.join(data_dir,"config.json") 
    variables = json.loads(open(con_file).read())
    totaldim=0
    Lower=np.array([])
    Upper=np.array([])
    for keys in variables:
        totaldim+=variables[keys]["size"]
        Lower= np.append(Lower,[float(variables[keys]["min"]) for n in range(0,variables[keys]["size"])])
        Upper= np.append(Upper,[float(variables[keys]["max"]) for n in range(0,variables[keys]["size"])])
    return totaldim,np.c_[Lower,Upper]




def Stats_3L(data_dir):
    #Goal: total original subspace dimensions; ordered list of variable locations/names for exchange; range of original variables
    con_file= os.path.join(data_dir,"config.json") ## json file keeping all original variables and their ranges/types
    totaldim=0
    vnames=[]
    Lower=np.array([])
    Upper=np.array([])
    #read in the configuration file denoting all of the original variables and ranges
    variables = json.loads(open(con_file).read())
    #first set of headers define the file to change with new variable settings
    for key in variables:
        #add the number of variable to total
        totaldim+=np.sum([vkey["size"] for vkey in variables[key]])
        #[subsection,var1,var2....]
        vn=[vkey["name"] for vkey in variables[key]]
        #expand lower range by size of the variable being reviewed
        Lower= np.append(Lower,[float(vkey["min"]) for vkey in variables[key] for n in range(0,int(vkey["size"]))])
        #expand upper range by size of the variable being reviewed
        Upper= np.append(Upper,[float(vkey["max"]) for vkey in variables[key] for n in range(0,int(vkey["size"]))])
        #combine all info on every filename
        vnames=[*vnames,[key,vn]]
    return vnames,totaldim,np.c_[Lower,Upper]

def Update_Variables(new_values,data_dir,src_dir,run_dir):
    # read in the json file for the destination choice in the current directory
    vnames,_,_=Stats_3L(data_dir)    
    for vkey in vnames:
        # read in the json file for the named file in the current directory
        t_fn=os.path.join(src_dir,vkey[0])
        u_fn=os.path.join(run_dir,vkey[0])
        #read the template file in
        dictionary=json.loads(open(t_fn).read())
        for dkey in dictionary:
            for ind in vkey[1]:
                dictionary[dkey][ind],new_values=new_values[0],new_values[1:]
        # save the updated file in the directory we use for this task
        with open(u_fn,'w') as fp:
            json.dump(dictionary, fp, indent=4)


def output_locs(scenario_file):
    dictionary=json.loads(open(scenario_file).read())
    output_dir=dictionary['Output controls']['output_dir_name']
    result_db=os.path.join(output_dir,dictionary['General simulation controls']['database_name']+'-Result.sqlite')

def config_convert_3L(save_fn,variables,ranges):
    r"""Creates a json configuration file for the specified variables in a 3-level nested structure. Should be
    used in cases where variables exist in seperate files in the simulation

    Args:
        save_fn (string): the filename that the configuration should be saved in
        variables (list): the variables [set1,set2,...] partitioned by file they are found in in the format of
                            [[file_name,variable_names],[file_name2,variable_names2]...]
        ranges (list): the minimum and maximum values each variable can take in in the format of
                            [[set1_min,set1_max],[set2_min,set2_max],...]
        
    Returns:
        a saved file in a 3-level configuration formation 


    Example:
        >>> variables = [["BloomingtonModeChoiceModel.json",["HBO_B_over65_tran", "NHB_B_ttime_tran"]],["BloomingdonDestinationChoiceModel.json",["NHB_B_peak_auto","HBO_B_ttime_tran","HBW_B_ttime_tran"]]]
        >>> ranges= [[np.ones(len(s[1]))*-10,np.ones(len(s[1]))*10]for s in variables]
        >>> save_fn='config.json'
    """
    #1st, check we have ranges for every variable
    if len(variables)!=len(ranges):
        raise ValueError("There are %d variable sets but %d range sets" % (len(variables),len(ranges)))
    elif not all([len(r)==2 for r in ranges]):
        raise ValueError("Not all range sets have a minimum and maximum range")
    elif not all([len(r)==len(variables[s][1]) for s in range(len(ranges)) for r in ranges[s]]):
        raise ValueError("Not enough range values for the number of variables in each set")

    #create nested structure
    newlines={}
    for i in range(len(variables)):
        newlines[variables[i][0]]=[]
        for v in range(len(variables[i][1])):
            newlines[variables[i][0]].append({
                    "name":variables[i][1][v],
                    "type":"float",
                    "min": str(ranges[i][0][v]),
                    "max":str(ranges[i][1][v]),
                    "size":1
                })              
    with open(save_fn,'w+') as fp:
        json.dump(newlines, fp, indent=4)
