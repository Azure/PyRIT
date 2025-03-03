# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import rpyc
import time
import socket

from typing import Callable, Optional
from threading import Thread, Semaphore, Event
from pyrit.models import PromptRequestPiece, Score

DEFAULT_PORT = 18812

class RPCClient:
    def __init__(self, callback_disconnected: Optional[Callable] = None):
        self.__c = None
        self.__bgsrv = None
        
        self.__ping_thread = None
        self.__bgsrv_thread = None
        self.__is_running = False

        self.__shutdown_event = None
        self.__prompt_received_sem = None

        self.__prompt_received = None
        self.__callback_disconnected = callback_disconnected

    def start(self):
        # Check if the port is open
        self.__wait_for_server_avaible()
        self.__prompt_received_sem = Semaphore(0)        
        
        self.__c = rpyc.connect("localhost", DEFAULT_PORT, config={'allow_public_attrs': True})
        self.__is_running = True
        self.__shutdown_event = Event()
        self.__bgsrv_thread = Thread(target=self.__bgsrv_lifecycle)
        self.__bgsrv_thread.start()

    def wait_for_prompt(self) -> PromptRequestPiece:
        self.__prompt_received_sem.acquire()
        return self.__prompt_received
    
    def send_prompt_response(self, response: bool):
        score = Score(
            score_value=str(response),
            score_type="true_false",
            score_category="safety",
            score_value_description="Safe" if response else "Unsafe",
            score_rationale="The prompt is safe" if response else "The prompt is unsafe",
            score_metadata={"prompt_target_identifier": self.__prompt_received.prompt_target_identifier},
            prompt_request_response_id=self.__prompt_received.conversation_id
        )
        self.__c.root.receive_score(score)

    def __wait_for_server_avaible(self):
        # Wait for the server to be available
        while not self.__is_server_running():
            print("Server is not running. Waiting for server to start...")
            time.sleep(1)

    def stop(self):
        """
        Stop the client.
        """
        # Send a signal to the thread to stop
        self.__shutdown_event.set()
    
    def reconnect(self):
        """
        Reconnect to the server.
        """
        self.stop()
        print("Reconnecting to server...")
        self.start()

    def __receive_prompt(self, prompt_request: PromptRequestPiece, task: Optional[str] = None):
        print(f"Received prompt: {prompt_request}")
        self.__prompt_received = prompt_request
        self.__prompt_received_sem.release()

    def __ping(self):
        try:
            while self.__is_running:
                self.__c.root.receive_ping()
                time.sleep(1.5)
        except EOFError:
            print("Connection closed")
            if self.__callback_disconnected is not None:
                self.__callback_disconnected()

    def __bgsrv_lifecycle(self):
        self.__bgsrv = rpyc.BgServingThread(self.__c)
        self.__ping_thread = Thread(target=self.__ping)
        self.__ping_thread.start()
        
        # Register callback
        self.__c.root.callback_score_prompt(self.__receive_prompt)

        # Wait for the server to be disconnected
        self.__shutdown_event.wait()

        self.__is_running = False
        self.__ping_thread.join()
        
        # Avoid calling stop() twice if the server is already stopped. This can happen if the server is stopped
        # by the ping request.
        if self.__bgsrv._active:
            self.__bgsrv.stop()
    
    def __is_server_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', DEFAULT_PORT)) == 0