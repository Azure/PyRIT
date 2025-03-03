# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import logging

from typing import Callable, Optional
from threading import Thread, Semaphore

from pyrit.ui.app import is_app_running, launch_app
from pyrit.models import Score, PromptRequestPiece


DEFAULT_PORT = 18812

logger = logging.getLogger(__name__)

# Exceptions
class RPCAppException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class RPCAlreadyRunningException(RPCAppException):
    """
    This exception is thrown when an RPC server is already running and the user tries to start another one.
    """
    def __init__(self):
        super().__init__("RPC server is already running.")

class RPCClientNotReadyException(RPCAppException):
    """
    This exception is thrown when the RPC client is not ready to receive messages.
    """
    def __init__(self):
        super().__init__("RPC client is not ready.")

class RPCServerStoppedException(RPCAppException):
    """
    This exception is thrown when the RPC server is stopped.
    """
    def __init__(self):
        super().__init__("RPC server is stopped.")


# RPC Server
class AppRPCServer:
    import rpyc
    # RPC Service
    class RPCService(rpyc.Service):
        """
        RPC service is the service that RPyC is using. RPC (Remote Procedure Call) is a way to interact with code that
        is hosted in another process or on an other machine. RPyC is a library that implements RPC and we are using to
        exchange information between PyRIT's main process and Gradio's process. This way the interface is
        independent of which PyRIT code is running the process.
        """
        def __init__(self, score_received_sem: Semaphore, client_ready_sem: Semaphore):
            super().__init__()
            self.__callback_score_prompt = None
            self.__last_ping = None
            self.__scores_received = []
            self.__score_received_sem = score_received_sem
            self.__client_ready_sem = client_ready_sem
        
        def on_connect(self, conn):
            logger.info("Client connected")

        def on_disconnect(self, conn):
            logger.info("Client disconnected")

        def exposed_receive_score(self, score: Score):
            logger.info(f"Score received: {score}")
            self.__scores_received.append(score)
            self.__score_received_sem.release()

        def exposed_receive_ping(self):
            # A ping should be received every 2s from the client. If a client misses a ping then the server should stoped
            self.__last_ping = time.time()
            logger.debug("Ping received")

        def exposed_callback_score_prompt(self, callback: Callable[[PromptRequestPiece, Optional[str]], None]):
            self.__callback_score_prompt = callback
            self.__client_ready_sem.release()
        
        def is_client_ready(self):
            if self.__callback_score_prompt is None:
                return False
            return True
        
        def send_score_prompt(self, prompt: PromptRequestPiece, task: Optional[str] = None):
            if not self.is_client_ready():
                raise RPCClientNotReadyException()
            self.__callback_score_prompt(prompt, task)

        def is_ping_missed(self):
            if self.__last_ping is None:
                return False
            
            return time.time() - self.__last_ping > 2
        
        def pop_score_received(self) -> Score | None:
            try:
                return self.__scores_received.pop()
            except IndexError:
                return None



    def __init__(self, open_browser: bool = False):
        self.__server = None
        self.__server_thread = None
        self.__rpc_service = None
        self.__is_alive_thread = None
        self.__is_alive_stop = False
        self.__score_received_sem = None
        self.__client_ready_sem = None
        self.__server_is_running = False
        self.__open_browser = open_browser

    def start(self):
        """
        Attempt to start the RPC server. If the server is already running, this method will throw an exception.
        """
        
        # Check if the server is already running by checking if the port is already in use.
        # If the port is already in use, throw an exception.
        if self.__is_instance_running():
            raise RPCAlreadyRunningException()
        
        self.__score_received_sem = Semaphore(0)
        self.__client_ready_sem = Semaphore(0)

        # Start the RPC server.
        self.__rpc_service = self.RPCService(self.__score_received_sem, self.__client_ready_sem)
        self.__server = self.rpyc.ThreadedServer(self.__rpc_service, port=DEFAULT_PORT, protocol_config={"allow_all_attrs": True})
        self.__server_thread = Thread(target=self.__server.start)
        self.__server_thread.start()

        # Start a thread to check if the client is still alive
        self.__is_alive_stop = False
        self.__is_alive_thread = Thread(target=self.__is_alive)
        self.__is_alive_thread.start()

        self.__server_is_running = True

        logger.info("RPC server started")

        if not is_app_running():
            logger.info("Launching Gradio UI")
            launch_app(open_browser=self.__open_browser)
        else:
            logger.info("Gradio UI is already running. Will not launch another instance.")

    def stop(self):
        """
        Stop the RPC server and free up the listening port.
        """
        self.stop_request()
        if self.__server is not None:
            self.__server_thread.join()


        if self.__is_alive_thread is not None:
            self.__is_alive_thread.join()
        
        logger.info("RPC server stopped")

    def stop_request(self):
        """
        Request the RPC server to stop. This method is does not block while waiting for the server to stop.
        """

        logger.info("RPC server stopping")
        if self.__server is not None:
            self.__server.close()
            self.__server = None


        if self.__is_alive_thread is not None:
            self.__is_alive_stop = True

        self.__server_is_running = False

        self.__client_ready_sem.release()
        self.__score_received_sem.release()

    def send_score_prompt(self, prompt: PromptRequestPiece, task: Optional[str] = None):
        """
        Send a score prompt to the client.
        """
        if self.__rpc_service is None:
            raise RPCAppException("RPC server is not running.")

        self.__rpc_service.send_score_prompt(prompt, task)

    def wait_for_score(self) -> Score:
        """
        Wait for the client to send a score. Should always return a score, but if the synchronisation fails it will return None.
        """
        if self.__score_received_sem is None or self.__rpc_service is None:
            raise RPCAppException("RPC server is not running.")

        self.__score_received_sem.acquire()
        if not self.__server_is_running:
            raise RPCServerStoppedException()
        
        score = self.__rpc_service.pop_score_received()
        if score is None:
            return None

        self.__client_ready_sem.release()
        return score
    
    def wait_for_client(self):
        """
        Wait for the client to be ready to receive messages.
        """
        if self.__client_ready_sem is None:
            raise RPCAppException("RPC server is not running.")

        
        logger.info("Waiting for client to be ready")
        self.__client_ready_sem.acquire()

        if not self.__server_is_running:
            raise RPCServerStoppedException()

        logger.info("Client is ready")

    def __is_instance_running(self):
        """
        Check if the RPC server is running.
        """
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', DEFAULT_PORT)) == 0

    def __is_alive(self):
        """
        Check if a ping has been missed. If a ping has been missed, stop the server.
        """
        while not self.__is_alive_stop:
            if self.__rpc_service.is_ping_missed():
                logger.error("Ping missed. Stopping server.")
                self.stop_request()
                break
            time.sleep(1)