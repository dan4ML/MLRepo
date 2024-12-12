from OraDBConn import OracleDBConnector
import torch
import os
import platform

class DBTransactions:    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dbName = None
        self.dbUser = None
        self.dbPW = None
        self.dbHost = None
        self.dbPort = None
        self.dbServiceName = None
        self.envConfig()
        self.readConfigFile()
        
    def envConfig(self):
        self.os_type = platform.system()

        if self.os_type == "Windows":
            self.pathToConfigFile = "windows/path/to/config_file"
        else:
            self.pathToConfigFile = "linux/path/to/config_file"

        print(f"Operating System: {self.os_type}, Config: {self.pathToConfigFile}")

    def readConfigFile(self):
        with open(self.pathToConfigFile, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments and empty lines
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'dbName':
                        self.dbName = value
                    elif key == 'dbUser':
                        self.dbUser = value
                    elif key == 'dbPW':
                        self.dbPW = value
                    elif key == 'dbHost':
                        self.dbHost = value
                    elif key == 'dbPort':
                        self.dbPort = value
                    elif key == 'dbServiceName':
                        self.dbServiceName = value  
                        
        #print("dbName: ",self.dbName," dbUser: ",self.dbUser, " dbPW: ",self.dbPW," dbHost: ",self.dbHost,"dbPort: ",self.dbPort,"servNam:",self.dbServiceName)                

        
    def executeQuery(self, query):
        print("query: ",query)
        
        dbConnObj = OracleDBConnector(self.dbName, self.dbUser, self.dbPW, self.dbHost, self.dbPort, self.dbServiceName)
        
        results =dbConnObj.executeQuery(query)
        
        return results
