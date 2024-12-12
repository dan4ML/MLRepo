import cx_Oracle
import os


class OracleDBConnector:
    def __init__(self, dbName, dbUser, dbPW, dbHost, dbPort, dbServiceName):
        
        self.dbName         = dbName
        self.dbUser         = dbUser
        self.dbPW           = dbPW
        self.dbHost         = dbHost
        self.dbPort         = dbPort
        self.dbServiceName  = dbServiceName
        self.dsn = cx_Oracle.makedsn(self.dbHost, self.dbPort, service_name=self.dbServiceName)
      
    def executeQuery(self, query):
        connection = None
        cursor = None
        try:
            # Connect to the Oracle database using the DSN
            connection = cx_Oracle.connect(user=self.dbUser, password=self.dbPW, dsn=self.dsn)
            cursor = connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except cx_Oracle.DatabaseError as e:
            print(f"Database error: {e}")
            raise e
        finally:
            # Safely close the cursor and connection
            if cursor:
                cursor.close()
            if connection:
                connection.close()
