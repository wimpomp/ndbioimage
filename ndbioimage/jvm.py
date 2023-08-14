from pathlib import Path
from urllib import request

try:
    class JVM:
        """ There can be only one java virtual machine per python process,
            so this is a singleton class to manage the jvm.
        """
        _instance = None
        vm_started = False
        vm_killed = False
        success = True

        def __new__(cls, *args):
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

        def __init__(self, jars=None):
            if not self.vm_started and not self.vm_killed:
                jarpath = Path(__file__).parent / 'jars'
                if jars is None:
                    jars = {}
                for jar, src in jars.items():
                    if not (jarpath / jar).exists():
                        JVM.download(src, jarpath / jar)
                classpath = [str(jarpath / jar) for jar in jars.keys()]

                import jpype
                jpype.startJVM(classpath=classpath)
                import jpype.imports
                from loci.common import DebugTools
                from loci.formats import ImageReader
                from loci.formats import ChannelSeparator
                from loci.formats import FormatTools
                from loci.formats import MetadataTools

                DebugTools.setRootLevel("ERROR")
                self.vm_started = True
                self.image_reader = ImageReader
                self.channel_separator = ChannelSeparator
                self.format_tools = FormatTools
                self.metadata_tools = MetadataTools

            if self.vm_killed:
                raise Exception('The JVM was killed before, and cannot be restarted in this Python process.')

        @staticmethod
        def download(src, dest):
            print(f'Downloading {dest.name} to {dest}.')
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(request.urlopen(src).read())

        def kill_vm(self):
            if self.vm_started and not self.vm_killed:
                import jpype
                jpype.shutdownJVM()
            self.vm_started = False
            self.vm_killed = True

except ImportError:
    JVM = None
