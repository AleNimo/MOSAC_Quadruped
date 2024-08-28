import socket
import numpy as np

class CoppeliaSocket:
    def __init__(self, obs_sp_size, port=57175):
        ''' Creates an object that controls the communications with the CoppeliaSim simulator '''
        self.__obs_sp_size = obs_sp_size     # Observation space size

        print("Waiting socket connection", flush=True)
        # Establish socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            s.listen()
            self.__coppelia_socket, addr = s.accept()
            print('Connected by', addr, flush=True)

        # Discard first observed state
        _ = [float(self.__coppelia_socket.recv(10).decode()) for idx in range(obs_sp_size)]


    def reset(self, obs):
        ''' Reset the simulation environment to the requested position and return the new state '''
        # Get the socket
        coppelia_socket = self.__coppelia_socket

        # Send the reset command and the requested position
        coppelia_socket.sendall("RESET".encode())
        #print(coppelia_socket.recv(5).decode())
        for data in obs:
            data = "{0:010.6f}".format(data)
            #print("Reset:",data)
            coppelia_socket.sendall(data.encode())

        # Receive and return the new observed state
        next_obs = np.array([float(coppelia_socket.recv(10).decode()) for idx in range(self.__obs_sp_size)])
        #print("Reset:",next_obs)
        return next_obs

    def act(self, act):
        ''' Take the requested action in the simulation and obtain the new state '''
        #print(act)
        # Get the socket
        coppelia_socket = self.__coppelia_socket

        # Send the act command and the requested action
        coppelia_socket.sendall("ACT__".encode())
        #print(coppelia_socket.recv(5).decode())
        for data in np.clip(act, -1.0, 1.0):
            data = "{0:010.6f}".format(data)
            #print("Act:", data)
            coppelia_socket.sendall(data.encode())

        # Receive and return the next observed state
        next_obs = np.array([float(coppelia_socket.recv(10).decode()) for idx in range(self.__obs_sp_size)])
        #print("Act:", next_obs)
        return next_obs

    def change_mode(self, mode):
        ''' Toggle between joint control and direction control modes '''
        coppelia_socket = self.__coppelia_socket

        # Send the act command and the requested action
        coppelia_socket.sendall("MODE_".encode())
        #print(coppelia_socket.recv(5).decode())
        data = "{0:010.6f}".format(mode)
        coppelia_socket.sendall(data.encode())

        # Discard first observed state
        _ = [float(self.__coppelia_socket.recv(10).decode()) for idx in range(self.__obs_sp_size)]
