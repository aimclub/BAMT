# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Nearly all of the functions of this library require key indexing, which means it deals with dictionaries internally. This module deals with loading dictionaries and handles automatically converting from python-style dictionaries to condensed (no excess white space) JSON-style dictionaries.

'''
import sys
import json
import string

class Dictionary(object):
    '''
    This class represents a JSON-style, key-indexable dictionary of data. It contains the attribute *alldata* and the method *dictload*. 
    '''

    def __init__(self):
        self.alldata = None
        '''An internal representation of a key-indexable dictionary.'''

    def dictload(self, path):
        '''
        Load a dictionary from a JSON-like text in a text file located at *path* into the attribute *alldata*.
        
        In order for this function to execute successfully, the text file must have the proper formatting, particularly with regard to quotation marks. See :doc:`unittestdict` for an example. Specifically, the function can get rid of excess whitespace, convert ``.x`` to ``0.x`` in decimals, and convert ``None`` to ``null``, but nothing else.

        Arguments:
            
            1. *path* -- Path to the text file (e.g. "mydictionary.txt")
        
        Attributes modified: 
        
            1. *alldata* -- The entire loaded dictionary.
        
        The function also returns an error if nothing was loaded into *alldata*.

        '''
        f = open(path, 'r')
        ftext = f.read() 
        assert (ftext and isinstance(ftext, str)), "Input file is empty or could not be read."


        # alter for json input, if necessary
        loaded = False
        try:
            self.alldata = json.loads(ftext)
            loaded = True
        except ValueError:
            pass

        if not loaded:
            try: 
                ftext = ftext.translate(None, '\t\n ')
                ftext = ftext.replace(':', ': ')
                ftext = ftext.replace(',', ', ')
                ftext = ftext.replace('None', 'null')
                ftext = ftext.replace('.', '0.')
                self.alldata = json.loads(ftext)
            except ValueError:
                raise (ValueError, "Convert to JSON from input file failed. Check formatting.")
        f.close()
        
        assert isinstance(self.alldata, dict), "In method dictload, path did not direct to a proper text file."
        
        
        
