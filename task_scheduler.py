def get_the_values(filename):
    crt = [0 for i in range(16)]
    task_matrix = [[0 for j in range(16)] for i in range(512)]
    file = open(filename,"r")
    i = 0
    j = 0
    for k,l in enumerate(file):
        if k!=0 and l!= '\n':
            task_matrix[i][j] = float(l)
            j = (j+1)%16
            if j == 0:
                i = (i+1)
                
    return crt, task_matrix
class TaskScheduler:
    def opportunistic_load_balancing(self, crt, task_matrix):
    
        #find the min crt of the clouds
        new_crt = list(crt)
        for task_no,task in enumerate(task_matrix): #as soon as the task approaches 
            
#             print(f"******************** {task_no+1} *********************")
            min_crt_index = -1
            min_crt_val = 10**20
            for i,val in enumerate(new_crt):
                
                if min_crt_val > val:
                    min_crt_val = val
                    min_crt_index = i
            #now assign the task to the resepective cloud 
#             print(new_crt)
#             print(new_crt[min_crt_index]+task[min_crt_index], " = ", new_crt[min_crt_index], " + ",task[min_crt_index])
            new_crt[min_crt_index] += task[min_crt_index]
#             print("minimum crt value ",min_crt_val, "task assigned", task[min_crt_index])
#             print("current CRT's", new_crt)
#             print(f" Task {task_no+1} is allocated to cloud {min_crt_index+1}")
        
    
        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan
        
    
    def minimum_execution_time(self, crt, task_matrix):
    
        new_crt = list(crt)
        for task_no,task in enumerate(task_matrix):
#             print(f"******************** {task_no+1} *********************")
            min_met_index = -1
            min_met_val = 10**20
            for i, val in enumerate(task):
                if val < min_met_val:
                    min_met_val = val
                    min_met_index = i
#             print(new_crt[min_met_index]+task[min_met_index], " = ", new_crt[min_met_index], " + ",task[min_met_index])
            new_crt[min_met_index] += task[min_met_index]
#             print("minimum execution time",min_met_val, "task assigned", task[min_met_index])
#             print("current CRT's", new_crt)
#             print(f" Task {task_no+1} is allocated to cloud {min_met_index+1}")


        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan

    def minimum_completion_time(self, crt, task_matrix):

        new_crt = list(crt)
        for task_no, task in enumerate(task_matrix):
#             print(f"******************** {task_no+1} *********************")
            min_mct_index = -1
            min_mct_val = 10**20
            for i,val in enumerate(task):

                present_mct = task[i]+new_crt[i]
                print(present_mct," = ",task[i],"+", new_crt[i])
                if present_mct<min_mct_val:
                    min_mct_val = present_mct
                    min_mct_index = i
#             print(new_crt[min_mct_index]+task[min_mct_index], " = ", new_crt[min_mct_index], " + ",task[min_mct_index])
            new_crt[min_mct_index] += task[min_mct_index]
#             print("minimum completion time",min_mct_val, "task assigned", task[min_mct_index])
#             print("Current CRTs", new_crt)
#             print(f"Task {task_no+1} is allocated to Cloud {min_mct_index+1}")


        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan
    
    
    def k_percent_best(self, crt, task_matrix, k):
    
        no_of_clouds = len(crt)
        best_cloud_no = ((no_of_clouds*k)//100) #scout only this no of clouds 
#         print("No of clouds we have to consider ",best_cloud_no)
        new_crt = list(crt)
        for task_no, task in enumerate(task_matrix):
#             print(f"******************** {task_no+1} *********************")
            best_clouds = [[val, i] for i,val in enumerate(task)] #first onboard the tasks
#             print("All the clouds exectution times")
#             for i in best_clouds:
#                 print(i[0])
            best_clouds = sorted(best_clouds, key = lambda x : x[0]) 
            best_clouds = best_clouds[:best_cloud_no]#get the clouds that have min execution times

#             print("k best clouds based on execution times")
#             for i in best_clouds:
#                 print(i[0])
        
#             print("Completion times computation")
#             for i in best_clouds:
#                 print(i[0]+new_crt[i[1]]," = ", i[0]," + ", new_crt[i[1]])
            best_clouds = [[cloud[0]+new_crt[cloud[1]], cloud[1]] for cloud in best_clouds] #calc the min completion times
            best_clouds = sorted(best_clouds, key = lambda x:x[0])
            min_mct_index = best_clouds[0][1]
#             print("After adding CRTs")
#             for i in best_clouds:
#                 print(i[0])
#             print("minimum completiond time", best_clouds[0][0], "task assigned", min_mct_index)
            new_crt[min_mct_index] += task[min_mct_index]
#             print("Current CRTs", new_crt)
#             print(f"Task {task_no} is allocated to Cloud {min_mct_index}")


        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan
    
    def switching_algorithm(self, crt, task_matrix, low, high):
    
        new_crt = list(crt)
        present = 0
        for task_no, task in enumerate(task_matrix):
#             print(f"******************** {task_no} *********************")
            if max(new_crt) == 0:
                present_val = 0  
            else :

                present_val = min(new_crt)/max(new_crt)
#                 print("ratio = ", min(new_crt),"/",max(new_crt))
#             print(new_crt, present_val)
            if low<=present_val and present_val<=high: #apply mct
#                 print("Apply MCT",)
                min_mct_index = -1
                min_mct_val = 10**20
                for i, val in enumerate(task):
                    present_ct = task[i] + new_crt[i]
#                     print(present_ct," = ",task[i],"+", new_crt[i])
                    if present_ct < min_mct_val :
                        min_mct_val = present_ct
                        min_mct_index = i 
#                 print(new_crt[min_mct_index]+task[min_mct_index]," = ", new_crt[min_mct_index] ,"+", task[min_mct_index])
                new_crt[min_mct_index] += task[min_mct_index]
#                 print("minimum completion time",min_mct_val, "task assigned ", task[min_mct_index])
#                 print("Current CRTs", new_crt)
#                 print(f"Task {task_no+1} is allocated to Cloud {min_mct_index+1}")

            else : #apply for met
#                 print("Apply MET")
                min_met_index = -1
                min_met_val = 10**20
                for i,val in enumerate(task):
                    if val < min_met_val:
                        min_met_val =  val
                        min_met_index = i
#                 print("Minimum execution value",min_met_val)
#                 print(new_crt[min_met_index]+task[min_met_index]," = ", new_crt[min_met_index] ,"+", task[min_met_index])
                new_crt[min_met_index] += task[min_met_index]
#                 print("Current CRTs", new_crt) 
#                 print(f"Task {task_no+1} is allocated to Cloud {min_met_index+1}")


        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan

    def min_min_max(self, crt, task_matrix, algo = "min-min"):

        new_crt = list(crt)
        no_of_tasks = len(task_matrix)
        flag = False
        if algo == "min-max":
            flag = True
        tasks_to_be_executed = [i for i in range(no_of_tasks)]
        while no_of_tasks>0:
#             print(f"******************** {no_of_tasks} *********************")
            column_matrix = []
#             print("Tasks completion times computation")
            for i in tasks_to_be_executed:
                min_mct_index = -1
                min_mct_val = 10**20
                
                for j,val in enumerate(task_matrix[i]):
                    present_ct = task_matrix[i][j] + new_crt[j]
#                     print(task_matrix[i][j]," + ",new_crt[j]," = ",present_ct, end= ' ')
                    if present_ct < min_mct_val:
                        min_mct_val = present_ct
                        min_mct_index = j
#                 print()
                column_matrix.append([i, min_mct_val, min_mct_index]) # task i is assigned to cloud min_mct_index


            column_matrix = sorted(column_matrix, key = lambda x : x[1], reverse=flag) #sort w.r.t mcl val in the column matrix

#             print("The Column matrix obtained")
#             for i in column_matrix:
#                 print(i[1])
            no_of_tasks -= 1
            assigned = column_matrix[0]
            assigned_cloud = assigned[2]
#             print("Assigned cloud", assigned_cloud+1,"Task allocated ", assigned[0]+1)
            new_crt[assigned_cloud] += (assigned[1]-new_crt[assigned_cloud])
            executed_task = tasks_to_be_executed.index(assigned[0])
            tasks_to_be_executed.pop(executed_task)
#             print("Remaining tasks")
#             for i in tasks_to_be_executed:
#                 print(i+1, end= ' ')
#             print()
#             print("Current CRTs ", new_crt)
#             print(f"Task {assigned[0]+1} is assigned to Cloud {assigned_cloud+1}")


        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan
    
    def sufferage(self, crt, task_matrix):

        new_crt = list(crt)

        no_of_tasks = len(task_matrix)
        tasks_to_be_executed = [i for i in range(no_of_tasks)]
        while no_of_tasks>0:
#             print(f"******************** {no_of_tasks} *********************")
            sufferage_values = []

            for i in tasks_to_be_executed:
                sufferage_calc = [[val+new_crt[j], j] for j,val in enumerate(task_matrix[i])]
                sufferage_calc = sorted(sufferage_calc)
                sufferage_val = (sufferage_calc[1][0]) - (sufferage_calc[0][0])
                sufferage_values.append([i, sufferage_val,sufferage_calc[0][1]])
                #task no, sufferage val , cloud no

            sufferage_values = sorted(sufferage_values, key = lambda x:x[1], reverse=True)
#             print("Sufferages values")
#             for i in sufferage_values:
#                 print(i[1])
            assigned = sufferage_values[0]
#             print("Max sufferage value = ", assigned[1])
            task_no = assigned[0]
            assigned_cloud = assigned[2]
            new_crt[assigned_cloud] += (task_matrix[task_no][assigned_cloud])
            executed_task = tasks_to_be_executed.index(task_no)
            tasks_to_be_executed.pop(executed_task)

            no_of_tasks -= 1
#             print("Current CRTs ", new_crt)
#             print(f"Task {assigned[0]+1} is assigned to Cloud {assigned_cloud+1}")

        makespan = max(new_crt)
#         print(f"Makespan for this dataset is {makespan}")
        return makespan

    def get_makespans(self, crt, task_matrix,type_of_algos = "offline"):
        
        makespans = []
        if type_of_algos == "offline":
            makespans.append(self.min_min_max(crt, task_matrix, "min-min")) #for minimum
            makespans.append(self.min_min_max(crt, task_matrix, "min-max"))
            makespans.append(self.sufferage(crt, task_matrix))
        else :
            makespans.append(self.oppurtunistic_load_balancing(crt, task_matrix))
            makespans.append(self.minimum_completion_time(crt, task_matrix))
            makespans.append(self.minimum_execution_time(crt, task_matrix))
            makespans.append(self.switching_algorithm(crt, task_matrix, 0.6,0.9))
            makespans.append(self.k_percent_best(crt, task_matrix, 20))
        return makespans
        

if __name__ == "__main__":
    filenames = ["u_c_hihi.txt" ,"u_c_hilo.txt" ,"u_c_lohi.txt" ,"u_c_lolo.txt",
             "u_i_hihi.txt" ,"u_i_hilo.txt" , "u_i_lohi.txt" , "u_i_lolo.txt" ,
             "u_s_hihi.txt","u_s_hilo.txt" ,"u_s_lohi.txt", "u_s_lolo.txt",]

    task_scheduler = TaskScheduler()
    store = {}
    for filename in filenames:
        crt, task_matrix = get_the_values(filename)
        
        makespans = task_scheduler.get_makespans(crt = crt,task_matrix = task_matrix, type_of_algos = "offline")
        store[filename[:-4]] = makespans

    for dataset, makespans in store.items():
        print(dataset, makespans)