import abc
from pathlib import Path
from functions.general_functions import listified

"""
Message service to send messages to different outputs (ignore, textfile, std out...) without the need to specify this
hardcoded
"""


class BaseMessageService:
    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Abstract init

        Parameters
        ----------
        kwargs:
            Key-worded arguments for implementations

        """
        pass

    @abc.abstractmethod
    # method implementation sends the message
    def send_message(self, message, **kwargs):
        """
        Send a message to the service.

        Parameters
        ----------
        message: String
            The message to send
        kwargs:
            Key-worded arguments for implementations

        """
        raise NotImplementedError('Abstract Method!')


class SilentMessageService(BaseMessageService):
    def __init__(self, **kwargs):
        """
        Service that does not send anything (for clean code)

        Parameters
        ----------
        kwargs:
            None
        """
        super().__init__(**kwargs)

    def send_message(self, message, **kwargs):
        """
        Does nothing with the message

        Parameters
        ----------
        message : String
            The message to do nothing with
        kwargs:
            None

        """
        pass


class PrintMessageService(BaseMessageService):
    def __init__(self, **kwargs):
        """
        Prints messages on stdout

        Parameters
        ----------
        kwargs
            None
        """
        super().__init__(**kwargs)

    def send_message(self, message, **kwargs):
        """
        Prints message on std out

        Parameters
        ----------
        message: String
            The message to print

        kwargs:
            None

        """
        print(message)


class FileMessageService(BaseMessageService):
    def __init__(self, **kwargs):
        """
        Send a message to a file.

        Other Parameters
        ----------------
        fn: str
            The location of the file to be written to.
        line_end: str, default \n
            Appended at the end of each message.
        """
        super().__init__(**kwargs)
        self.fn = Path(kwargs['fn'])
        self.line_end = kwargs.get('line_end', '\n')
        Path(self.fn).parent.mkdir(parents=True, exist_ok=True)

    def send_message(self, message, **kwargs):
        with open(self.fn, 'a') as wf:
            wf.write(f'{message}{self.line_end}')


class CompositeMessageService(BaseMessageService):

    def __init__(self, **kwargs):
        """
        Send a message to multiple services

        Parameters
        ----------
        kwargs
            **msg_services** (Collection of) BaseMessageService, optional (default = SilentMessageService)
                The message services to forward to.
        """
        super().__init__(**kwargs)
        self.msg_services = listified(kwargs.pop('msg_services', SilentMessageService()), BaseMessageService)

    def add_msg_service(self, other):
        if isinstance(other, BaseMessageService):
            self.msg_services.append(other)
        else:
            raise TypeError('Given argument should be BaseMessageService')

    def send_message(self, message, **kwargs):
        """
        Send a message to all the services

        Parameters
        ----------
        message: String
            The message to forward
        kwargs:
            None

        """
        for msg_service in self.msg_services:
            msg_service.send_message(message, **kwargs)
