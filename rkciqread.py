import numpy as np
from collections import namedtuple
import struct
import os
import sys

class rkcfile(object):
    def __init__(self, filename, maxPulse=None):
        """ initialize. """

        # # open the file if file object not passed
        # if hasattr(filename, 'read'):
        #     fobj = filename
        # else:
        #     fobj = open(filename, 'rb')
        # self._fh = fobj
        UINT8 = 'B'
        INT16 = 'h'
        UINT16 = 'H'
        UINT32 = 'I'
        UINT64 = 'Q'
        SINGLE = 'f'
        DOUBLE = 'd'

        fobj = open(filename,'rb')
        self.constants=self.rkc_constant()
        self.header=self.rkc_header()
        self.header.preface = ''.join([chr(item) for item in fobj.read(self.constants.RKName)])
        self.header.buildNo = int(np.fromfile(fobj, dtype=np.uint32 , count=1))
        print('preface = '+self.header.preface+'   buildNo = '+str(self.header.buildNo)+'\n')
        if self.header.buildNo >= 5:
            self.header.dataType = int(np.fromfile(fobj, dtype=np.uint8 , count=1))
        if self.header.buildNo >= 2:
            if self.header.buildNo >= 6:
                # RKRadarDescOffset
                offset = self.constants.RKRadarDescOffset
            else:
                # RKName * (char) + (uint32_t)
                offset = self.constants.RKName + 4
        else:
            print('buildNo = '+str(self.header.buildNo)+' unexpected.\n')

        if self.header.buildNo >= 2:
            RKC_DESC_HEADER = (
                ('initFlags',UINT32,1),
                ('pulseCapacity',UINT32,1),
                ('pulseToRayRatio',UINT16,1),
                ('doNotUse',UINT16,1),
                ('healthNodeCount',UINT32,1),
                ('healthBufferDepth',UINT32,1),
                ('statusBufferDepth',UINT32,1),
                ('configBufferDepth',UINT32,1),
                ('positionBufferDepth',UINT32,1),
                ('pulseBufferDepth',UINT32,1),
                ('rayBufferDepth',UINT32,1),
                ('productBufferDepth',UINT32,1),
                ('controlCapacity',UINT32,1),
                ('waveformCalibrationCapacity',UINT32,1),
                ('healthNodeBufferSize',UINT64,1),
                ('healthBufferSize',UINT64,1),
                ('statusBufferSize',UINT64,1),
                ('configBufferSize',UINT64,1),
                ('positionBufferSize',UINT64,1),
                ('pulseBufferSize',UINT64,1),
                ('rayBufferSize',UINT64,1),
                ('productBufferSize',UINT64,1),
                ('pulseSmoothFactor',UINT32,1),
                ('pulseTicsPerSecond',UINT32,1),
                ('positionSmoothFactor',UINT32,1),
                ('positionTicsPerSecond',UINT32,1),
                ('positionLatency',DOUBLE,1),
                ('latitude',DOUBLE,1),
                ('longitude',DOUBLE,1),
                ('heading',SINGLE,1),
                ('radarHeight',SINGLE,1),
                ('wavelength',SINGLE,1),
                ('name_raw',str(self.constants.RKName)+'s',1),
                ('filePrefix_raw',str(self.constants.RKMaximumPrefixLength)+'s',1),
                ('dataPath_raw',str(self.constants.RKMaximumFolderPathLength)+'s',1)
                )
        elif self.header.buildNo == 1:
            RKC_DESC_HEADER = (
                ('initFlags',UINT32,1),
                ('pulseCapacity',UINT32,1),
                ('pulseToRayRatio',UINT32,1),
                ('healthNodeCount',UINT32,1),
                ('configBufferDepth',UINT32,1),
                ('positionBufferDepth',UINT32,1),
                ('pulseBufferDepth',UINT32,1),
                ('rayBufferDepth',UINT32,1),
                ('controlCount',UINT32,1),
                ('latitude',DOUBLE,1),
                ('longitude',DOUBLE,1),
                ('heading',SINGLE,1),
                ('radarHeight',SINGLE,1),
                ('wavelength',SINGLE,1),
                ('name_raw',str(self.constants.RKName)+'s',1),
                ('filePrefix_raw',str(self.constants.RKName)+'s',1),
                ('dataPath_raw',str(self.constants.RKMaximumPathLength)+'s',1)
                )
        else :
            print('buildNo = '+str(outer.header.buildNo)+' unexpected.\n')

        data=_unpack_rkc_structure(fobj,RKC_DESC_HEADER,offset)
        data.update({'name':data['name_raw'][0].decode('utf-8').replace('\x00','')})
        data.update({'filePrefix':data['filePrefix_raw'][0].decode('utf-8').replace('\x00','')})
        data.update({'dataPath':data['dataPath_raw'][0].decode('utf-8').replace('\x00','')})
        self.header.desc = namedtuple('desc', data.keys())(*data.values())

        if self.header.buildNo == 6:
            offset = self.constants.RKRadarDescOffset + self.constants.RKRadarDesc
            # self.header.config
                # %  Read in RKConfig
            RKC_CONFIG_HEADER=(
                ('i',UINT64,1),
                ('sweepElevation',SINGLE,1),
                ('sweepAzimuth',SINGLE,1),
                ('startMarker',UINT32,1),
                ('prt',str(self.constants.RKMaxFilterCount)+SINGLE,self.constants.RKMaxFilterCount),
                ('pw',str(self.constants.RKMaxFilterCount)+SINGLE,self.constants.RKMaxFilterCount),
                ('pulseGateCount',UINT32,1),
                ('pulseGateSize',SINGLE,1),
                ('transitionGateCount',UINT32,1),
                ('ringFilterGateCount',UINT32,1),
                ('waveformId',str(self.constants.RKMaxFilterCount)+UINT32,self.constants.RKMaxFilterCount),
                ('noise','2'+SINGLE,2),
                ('systemZCal','2'+SINGLE,2),
                ('systemDCal',SINGLE,1),
                ('systemPCal',SINGLE,1),
                ('ZCal',str(2*self.constants.RKMaxFilterCount)+SINGLE,2*self.constants.RKMaxFilterCount),
                ('DCal',str(self.constants.RKMaxFilterCount)+SINGLE,self.constants.RKMaxFilterCount),
                ('PCal',str(self.constants.RKMaxFilterCount)+SINGLE,self.constants.RKMaxFilterCount),
                ('SNRThreshold',SINGLE,1),
                ('SQIThreshold',SINGLE,1),
                ('waveformName',str(self.constants.RKName)+'s',1),
                ('trash','2'+UINT64,2),
                ('vcpDefinition',str(self.constants.RKMaximumCommandLength)+'s',1)
                )
            data=_unpack_rkc_structure(fobj,RKC_CONFIG_HEADER,offset,popfield=['trash'])
            # data.pop('trash', None)
            # size = _structure_size(RKC_CONFIG_HEADER)
            # fobj.seek(offset,0)
            # buf = fobj.read(size)
            # # print(struct.unpack('<4h',buf[0:8]))
            # data = _unpack_from_buf(buf, 0, RKC_CONFIG_HEADER)
            data.update({'waveformName':data['waveformName'][0].decode('utf-8').replace('\x00','')})
            data.update({'vcpDefinition':data['vcpDefinition'][0].decode('utf-8').replace('\x00','')})
            self.header.config = namedtuple('config', data.keys())(*data.values())
            offset = self.constants.RKFileHeader
            RKC_WAVEFORM_HEADER=(
                ('count',UINT8,1),
                ('depth',UINT32,1),
                ('type',UINT32,1),
                ('name','128s',1),
                ('fc',DOUBLE,1),
                ('fs',DOUBLE,1),
                ('filterCounts',str(self.constants.RKMaximumWaveformCount)+UINT8,self.constants.RKMaximumWaveformCount)
                )
            # size = _structure_size(RKC_WAVEFORM_HEADER)
            # fobj.seek(offset,0)
            # buf = fobj.read(size)
            # data = _unpack_from_buf(buf, 0, RKC_WAVEFORM_HEADER)
            data=_unpack_rkc_structure(fobj,RKC_WAVEFORM_HEADER,offset)
            data.update({'filterCounts':data['filterCounts'][0:data['count'][0]]})
            data.update({'name':data['name'][0].decode('utf-8').replace('\x00','')})
            self.header.waveform = namedtuple('waveform', data.keys())(*data.values())
            filters = []
            tones = []
            RKC_WAVEFORM_FILTER_HEADER=(
                ('name',UINT32,1),
                ('origin',UINT32,1),
                ('length',UINT32,1),
                ('inputOrigin',UINT32,1),
                ('outputOrigin',UINT32,1),
                ('maxDataLength',UINT32,1),
                ('subCarrierFrequency',SINGLE,1),
                ('sensitivityGain',SINGLE,1),
                ('filterGain',SINGLE,1),
                ('fullScale',SINGLE,1),
                ('lowerBoundFrequency',SINGLE,1),
                ('upperBoundFrequency',SINGLE,1),
                ('padding','16'+UINT8,16)
                )
            RKC_WAVEFORM_TONES_HEADER=(
                ('samples',str(2*self.header.waveform.depth[0])+SINGLE,2*self.header.waveform.depth[0]),
                ('iSamples',str(2*self.header.waveform.depth[0])+INT16,2*self.header.waveform.depth[0])
                )
            offset = offset + self.constants.RKWaveFileGlobalHeader
            for iw in range(self.header.waveform.count[0]):
                tmp=[]
                # print(self.header.waveform.filterCounts[iw])
                for ifc in range(self.header.waveform.filterCounts[iw]):
                    w=_unpack_rkc_structure(fobj,RKC_WAVEFORM_FILTER_HEADER,offset,repeat=self.header.waveform.filterCounts[iw],popfield=['padding'])
                    tmp.append(w)
                    offset = offset + self.constants.RKFilterAnchorSize
                filters.append(tmp)
                w2=_unpack_rkc_structure(fobj,RKC_WAVEFORM_TONES_HEADER,offset)
                depth=self.header.waveform.depth[0]
                offset = offset + 2 * depth * (4 + 2)
                x = np.reshape(np.array(w2['samples']),(depth,2))
                y = np.reshape(np.array(w2['iSamples']),(depth,2))
                gsamp={'samples':x[:,0]+1j*x[:,1],'iSamples':y[:,0]+1j*y[:,1]}
                # x = np.reshape(np.array(w2['samples']),(2,depth))
                # y = np.reshape(np.array(w2['iSamples']),(2,depth))
                # gsamp={'samples':x[0,:]+1j*x[1,:],'iSamples':y[0,:]+1j*y[1,:]}
                tones.append(gsamp)
            # print(dir(self.header.waveform))
            # self.header.waveform.filters = filters
            # self.header.waveform.tones = tones
            data.update({'filters':filters,'tones':tones})
            self.header.waveform = namedtuple('waveform', data.keys())(*data.values())
            # print(self.header.waveform)
            # print(self.header.waveform.tones)
            self.header.config=self.header.config._replace(pw = self.header.config.pw[self.header.waveform.filterCounts[0]],prt = np.asarray(self.header.config.prt)[np.asarray(self.header.config.prt) > 0])
            # self.header.config._replace(prt = np.asarray(self.header.config.prt)[np.asarray(self.header.config.prt) > 0])
        elif self.header.buildNo == 5:
            print('build=5')
        # header unfinish
        fobj.seek(offset + 28, 0)
        capacity=struct.unpack('<'+UINT32, fobj.read(struct.calcsize('<'+UINT32)))[0]
        gateCount=struct.unpack('<'+UINT32, fobj.read(struct.calcsize('<'+UINT32)))[0]
        downSampledGateCount=struct.unpack('<'+UINT32, fobj.read(struct.calcsize('<'+UINT32)))[0]
        print('gateCount = ',gateCount,' capacity = ',capacity,' downSampledGateCount = ', downSampledGateCount)
        print('data offset = ', offset)

        if self.header.dataType == 2:
                # % Compressed I/
            RKC_PULSE_MAP = (
            ('i',UINT64,1),
            ('n',UINT64,1),
            ('t',UINT64,1),
            ('s',UINT32,1),
            ('capacity',UINT32,1),
            ('gateCount',UINT32,1),
            ('downSampledGateCount',UINT32,1),
            ('marker',UINT32,1),
            ('pulseWidthSampleCount',UINT32,1),
            ('time_tv_sec',UINT64,1),
            ('time_tv_usec',UINT64,1),
            ('timeDouble',DOUBLE,1),
            ('rawAzimuth',str(4)+UINT8,4),
            ('rawElevation',str(4)+UINT8,4),
            ('configIndex',UINT16,1),
            ('configSubIndex',UINT16,1),
            ('azimuthBinIndex',UINT16,1),
            ('gateSizeMeters',SINGLE,1),
            ('elevationDegrees',SINGLE,1),
            ('azimuthDegrees',SINGLE,1),
            ('elevationVelocityDegreesPerSecond',SINGLE,1),
            ('azimuthVelocityDegreesPerSecond',SINGLE,1),
            ('iq',str(2*downSampledGateCount*2)+SINGLE,2*downSampledGateCount*2)
            )
        else:
            RKC_PULSE_MAP = (
            ('i',UINT64,1),
            ('n',UINT64,1),
            ('t',UINT64,1),
            ('s',UINT32,1),
            ('capacity',UINT32,1),
            ('gateCount',UINT32,1),
            ('downSampledGateCount',UINT32,1),
            ('marker',UINT32,1),
            ('pulseWidthSampleCount',UINT32,1),
            ('time_tv_sec',UINT64,1),
            ('time_tv_usec',UINT64,1),
            ('timeDouble',DOUBLE,1),
            ('rawAzimuth',str(4)+UINT8,4),
            ('rawElevation',str(4)+UINT8,4),
            ('configIndex',UINT16,1),
            ('configSubIndex',UINT16,1),
            ('azimuthBinIndex',UINT16,1),
            ('gateSizeMeters',SINGLE,1),
            ('elevationDegrees',SINGLE,1),
            ('azimuthDegrees',SINGLE,1),
            ('elevationVelocityDegreesPerSecond',SINGLE,1),
            ('azimuthVelocityDegreesPerSecond',SINGLE,1),
            ('iq',str(2*gateCount*2)+INT16,2*downSampledGateCount*2)
            )
        fobj.seek(offset, 0)
        if maxPulse is None:
            maxPulse = len(fobj.read())//_structure_size(RKC_PULSE_MAP)
        print('Reading ', maxPulse,' pulses ...')
        m = _unpack_rkc_structure(fobj,RKC_PULSE_MAP,offset,repeat=maxPulse)
        self.pulses = m
        if self.header.dataType == 1:
            dstr = 'raw'
        elif self.header.dataType == 2:
            dstr = 'compressed'
        else:
            dstr = 'unknown'
        self.header.dataType = dstr

    def getpulses(self):
        iqbuf=[]
        for ib in self.pulses:
            iqbuf.append(ib['iq'])
        iqbuf=np.asarray(iqbuf)
        iqbuf=np.reshape(iqbuf,(iqbuf.shape[0],2,iqbuf.shape[1]//4,2))
        iqpulse=iqbuf[:,:,:,0]+1j*iqbuf[:,:,:,1]
        iqpulse = np.moveaxis(iqpulse,[2,0,1],[0,1,2])
        # print(iqpulse[:,0,1])
        return iqpulse

    def getazimuth(self):
        azbuf=[]
        for ib in self.pulses:
            azbuf.append(ib['azimuthDegrees'])
        return np.squeeze(np.asarray(azbuf))

    def getelevation(self):
        elebuf=[]
        for ib in self.pulses:
            elebuf.append(ib['elevationDegrees'])
        return np.squeeze(np.asarray(elebuf))

    class rkc_constant(object):
        def __init__(self):
            self.RKName = 128
            self.RKFileHeader = 4096
            self.RKMaxMatchedFilterCount = 8
            self.RKFilterAnchorSize = 64
            self.RKMaximumStringLength = 4096
            self.RKMaximumPathLength = 1024
            self.RKMaximumPrefixLength = 8
            self.RKMaximumFolderPathLength = 768
            self.RKMaximumWaveformCount = 22
            self.RKMaximumFilterCount = 8
            self.RKRadarDescOffset = 256
            self.RKRadarDesc = 1072
            self.RKConfigV1 = 1441
            self.RKConfig = 1024
            self.RKMaximumCommandLength = 512
            self.RKMaxFilterCount = 8
            self.RKPulseHeader = 256
            self.RKWaveFileGlobalHeader = 512
    class rkc_header(object):
        def __init__(self):
            self.preface = None
            self.buildNo = None
            self.dataType = None
    # class rkc_desc(object):
    #     def __init__(self,fobj,offset,outer):
    #         UINT8 = 'B'
    #         INT16 = 'h'
    #         UINT16 = 'H'
    #         UINT32 = 'I'
    #         UINT64 = 'Q'
    #         SINGLE = 'f'
    #         DOUBLE = 'd'
    #         if outer.header.buildNo >= 2:
    #             RKC_DESC_HEADER = (
    #                 ('initFlags',UINT32,1),
    #                 ('pulseCapacity',UINT32,1),
    #                 ('pulseToRayRatio',UINT16,1),
    #                 ('doNotUse',UINT16,1),
    #                 ('healthNodeCount',UINT32,1),
    #                 ('healthBufferDepth',UINT32,1),
    #                 ('statusBufferDepth',UINT32,1),
    #                 ('configBufferDepth',UINT32,1),
    #                 ('positionBufferDepth',UINT32,1),
    #                 ('pulseBufferDepth',UINT32,1),
    #                 ('rayBufferDepth',UINT32,1),
    #                 ('productBufferDepth',UINT32,1),
    #                 ('controlCapacity',UINT32,1),
    #                 ('waveformCalibrationCapacity',UINT32,1),
    #                 ('healthNodeBufferSize',UINT64,1),
    #                 ('healthBufferSize',UINT64,1),
    #                 ('statusBufferSize',UINT64,1),
    #                 ('configBufferSize',UINT64,1),
    #                 ('positionBufferSize',UINT64,1),
    #                 ('pulseBufferSize',UINT64,1),
    #                 ('rayBufferSize',UINT64,1),
    #                 ('productBufferSize',UINT64,1),
    #                 ('pulseSmoothFactor',UINT32,1),
    #                 ('pulseTicsPerSecond',UINT32,1),
    #                 ('positionSmoothFactor',UINT32,1),
    #                 ('positionTicsPerSecond',UINT32,1),
    #                 ('positionLatency',DOUBLE,1),
    #                 ('latitude',DOUBLE,1),
    #                 ('longitude',DOUBLE,1),
    #                 ('heading',SINGLE,1),
    #                 ('radarHeight',SINGLE,1),
    #                 ('wavelength',SINGLE,1),
    #                 ('name_raw',str(outer.constants.RKName)+'s',1),
    #                 ('filePrefix_raw',str(outer.constants.RKMaximumPrefixLength)+'s',1),
    #                 ('dataPath_raw',str(outer.constants.RKMaximumFolderPathLength)+'s',1)
    #                 )
    #         elif outer.header.buildNo == 1:
    #             RKC_DESC_HEADER = (
    #                 ('initFlags',UINT32,1),
    #                 ('pulseCapacity',UINT32,1),
    #                 ('pulseToRayRatio',UINT32,1),
    #                 ('healthNodeCount',UINT32,1),
    #                 ('configBufferDepth',UINT32,1),
    #                 ('positionBufferDepth',UINT32,1),
    #                 ('pulseBufferDepth',UINT32,1),
    #                 ('rayBufferDepth',UINT32,1),
    #                 ('controlCount',UINT32,1),
    #                 ('latitude',DOUBLE,1),
    #                 ('longitude',DOUBLE,1),
    #                 ('heading',SINGLE,1),
    #                 ('radarHeight',SINGLE,1),
    #                 ('wavelength',SINGLE,1),
    #                 ('name_raw',str(outer.constants.RKName)+'s',1),
    #                 ('filePrefix_raw',str(outer.constants.RKName)+'s',1),
    #                 ('dataPath_raw',str(outer.constants.RKMaximumPathLength)+'s',1)
    #                 )
    #         else :
    #             print('buildNo = '+str(outer.header.buildNo)+' unexpected.\n')
    #         size = outer._structure_size(RKC_DESC_HEADER)
    #         fobj.seek(offset,0)
    #         buf = fobj.read(size)
    #         data = outer._unpack_from_buf(buf, 0, RKC_DESC_HEADER)
    #         data.update({'name':data['name_raw'][0].decode('utf-8').replace('\x00','')})
    #         data.update({'filePrefix':data['filePrefix_raw'][0].decode('utf-8').replace('\x00','')})
    #         data.update({'dataPath':data['dataPath_raw'][0].decode('utf-8').replace('\x00','')})
    #         self.data = namedtuple('desc', data.keys())(*data.values())

def _structure_size(structure):
    """ Find the size of a structure in bytes. """
    return struct.calcsize('<'+''.join([i[1] for i in structure]))


def _unpack_from_buf(buf, pos, structure):
    """ Unpack a structure from a buffer. """
    size = _structure_size(structure)
    return _unpack_structure(buf[pos:pos + size], structure)


def _unpack_structure(string, structure):
    """ Unpack a structure from a string """
    fmt = '<'+''.join([i[1] for i in structure])  # UF is big-endian
    lst = struct.unpack(fmt, string)
    n = 0
    ost=[]
    for i in structure:
        ost.append(list(lst[n:n+i[2]]))
        n=n+i[2]
    return dict(zip([i[0] for i in structure], ost))

def _unpack_rkc_structure(fobj,header,offset,repeat=1,popfield=[]):
    size = _structure_size(header)
    fobj.seek(offset,0)
    if repeat==1:
        buf = fobj.read(size)
        out_dict=_unpack_from_buf(buf, 0, header)
        for ip in popfield:
            out_dict.pop(ip, None)
        return out_dict
    else:
        tmp=[]
        for ir in range(repeat):
            buf = fobj.read(size)
            out_dict=_unpack_from_buf(buf, 0, header)
            for ip in popfield:
                out_dict.pop(ip, None)
            tmp.append(out_dict)
        return tmp

# def samSt(St,angar,dang=1):
#     ng = St.shape[0]
#     nc = St.shape[2]
#     nang = int(np.ptp(angar)//dang)
#     ns = int(St.shape[1]//nang)
#     print('St :: nang = ',nang,'ns = ',ns)
#     St=St[:,:int(nang*ns),:]
#     return np.reshape(St,(ng,nang,ns,nc)), angar[ns//2:-ns//2:ns]
def samSt(St,angar,dang=0.5,angswath=3):
    ng = St.shape[0]
    nray = St.shape[1]
    nc = St.shape[2]
    nsw = int(np.ptp(angar)//angswath)
    ns = int(nray//nsw)
    npointing = int(np.ptp(angar)//dang)
    angspacing = int(nray//npointing)
    ang_start = int(ns//2)
    nang = int((nray-ns)//angspacing)
    buf=np.zeros((ng,nang,ns,nc),dtype=St.dtype)
    angbuf=np.zeros(nang,dtype=float)
    for ia in range(nang):
        buf[:,ia,:,:] = St[:,ia*angspacing:ia*angspacing+ns,:]
        angbuf[ia] = angar[ang_start+ia*angspacing]
    print('St :: nang = ',nang,'ns = ',ns)
    # St=St[:,:int(nang*ns),:]
    return buf, angbuf

def samSt_select(St,angar,selectang,angswath=3):
    mask=np.where(np.logical_and(angar>selectang-angswath/2 , angar<selectang+angswath/2))
    slc=slice(np.min(mask),np.max(mask)+1)
    buf=St[:,slc,:]
    buf=buf[:,np.newaxis,:,:]
    return buf

def MOMZ(St,dr,KDPsmooth=20):
    SH = St[:,:,:,0]*np.conj(St[:,:,:,0])
    SV = St[:,:,:,1]*np.conj(St[:,:,:,1])
    SX = St[:,:,:,0]*np.conj(St[:,:,:,1])
    ZH = np.mean(SH,axis=2)
    ZV = np.mean(SV,axis=2)
    ZX = np.mean(SX,axis=2)
    ZDR = ZH/ZV
    RHO = np.abs(ZX)/np.sqrt(ZV*ZH)
    PHI = np.angle(ZX/np.sqrt(ZV*ZH))*180/np.pi
    buf=np.zeros(PHI.shape)
    for ir in range(KDPsmooth):
        buf[KDPsmooth//2:-KDPsmooth+KDPsmooth//2,:]=buf[KDPsmooth//2:-KDPsmooth+KDPsmooth//2,:]+PHI[ir:-KDPsmooth+ir,:]
    PHI = buf/KDPsmooth
    KDP = np.concatenate((np.zeros((1,PHI.shape[1])),(PHI[1:,:]-PHI[:-1,:])/dr*1e3/2))
    data={'ZH':ZH,'ZV':ZV,'ZDR':ZDR,'RHO':RHO,'PHI':PHI,'KDP':KDP}
    return namedtuple('MOMENT', data.keys())(*data.values())

def sZf(St,zero_pad=64,bootstrap=16,nstrap=20,window = None):
    if window is None:
        wd = np.ones(bootstrap)[np.newaxis,np.newaxis,:,np.newaxis]
    elif window == 'Rect':
        wd = np.ones(bootstrap)[np.newaxis,np.newaxis,:,np.newaxis]
    elif window == 'Hanning':
        wd = np.hanning(bootstrap)[np.newaxis,np.newaxis,:,np.newaxis]
    elif window == 'Blackman':
        wd = np.blackman(bootstrap)[np.newaxis,np.newaxis,:,np.newaxis]
    R0=np.mean(St*np.conj(St),axis=2)
    ex_width = int(np.round((0.5 - np.sqrt(np.mean(wd**2))*0.5)*St.shape[2]))
    Cx_L = (St[:,:,0,0]/St[:,:,-1,0]+St[:,:,0,1]/St[:,:,-1,1])/2
    Cx_R = (St[:,:,-1,0]/St[:,:,0,0]+St[:,:,-1,1]/St[:,:,0,1])/2
    StL = St[:,:,-ex_width:-1,:]*Cx_L[:,:,np.newaxis,np.newaxis]
    StR = St[:,:,2:ex_width+1,:]*Cx_R[:,:,np.newaxis,np.newaxis]
    ex_St = np.concatenate((StL,St,StR),axis=2)
    SH=np.zeros((St.shape[0],St.shape[1],zero_pad),dtype=St.dtype)
    SV=SH.copy()
    SX=SH.copy()
    np.random.seed(123)
    for istart in np.round(np.random.random(nstrap)*(ex_St.shape[2]-bootstrap)).astype(int):
        print('boostrap ',istart,'/',ex_St.shape[2])
        sys.stdout.flush()
        tR0 = np.mean(ex_St[:,:,istart:istart+bootstrap,:]*np.conj(ex_St[:,:,istart:istart+bootstrap,:]),axis=2)[:,:,np.newaxis,:]
        CR0=np.sqrt(R0[:,:,np.newaxis,:]/tR0)
    # np.random.seed(123)
    # for istart in np.round(np.random.random(nstrap)*(ex_St.shape[2]-bootstrap)).astype(int):
    #     print('boostrap ',istart,'/',ex_St.shape[2])
    #     sys.stdout.flush()
        # buf[range, pointing angle, velocity, channel]
        buf=np.fft.fftshift(np.fft.fft(wd*CR0*ex_St[:,:,istart:istart+bootstrap,:],n=zero_pad,axis=2),axes=2)
        SH = SH+buf[:,:,:,0]*np.conj(buf[:,:,:,0])
        SV = SV+buf[:,:,:,1]*np.conj(buf[:,:,:,1])
        SX = SX+buf[:,:,:,0]*np.conj(buf[:,:,:,1])
    alpha = np.mean(abs(wd)**2)
    SH=SH/bootstrap/alpha/nstrap
    SV=SV/bootstrap/alpha/nstrap
    SX=SX/bootstrap/alpha/nstrap

    return SH,SV,SX

def range_avg(ar,bootstrap=8):
    ar=ar[:ar.shape[0]//bootstrap*bootstrap,:,:]
    outar = np.zeros(ar[::bootstrap,:,:].shape,dtype=ar.dtype)
    for ix in range(bootstrap):
        outar=ar[ix::bootstrap,:,:]+outar
    return outar/bootstrap

def phi_smooth(ax,n):
    buf=np.zeros(ax.shape,dtype=ax.dtype)
    for ir in range(n):
        buf[n//2:-n+n//2,:,:]=buf[n//2:-n+n//2,:,:]+ax[ir:-n+ir,:,:]
    return buf/n

def phi_smooth_V(ax,n):
    buf=np.zeros(ax.shape,dtype=ax.dtype)
    for ir in range(n):
        buf[:,:,n//2:-n+n//2]=buf[:,:,n//2:-n+n//2]+ax[:,:,ir:-n+ir]
    return buf/n

def radar_navigation(radar_dict):
    radar_lon=radar_dict['rlon']
    radar_lat=radar_dict['rlat']
    radar_theta=radar_dict['theta']
    radar_range=radar_dict['range']
    radar_az=radar_dict['az']
    # rng, az = np.meshgrid(np.arange(radar_gate_start,radar_gate_sp*(radar_ngate-0.1)+radar_gate_start,radar_gate_sp), np.arange(radar_azm_start,radar_azm_sp*(radar_nray-0.1)+radar_azm_start,radar_azm_sp))
    rng, az = np.meshgrid(radar_range, radar_az)
    rng, ele = np.meshgrid(radar_range, radar_theta)
    theta_e = ele * np.pi / 180.0       # elevation angle in radians.
    theta_a = az * np.pi / 180.0        # azimuth angle in radians.
    Re = 6371.0 * 1000.0 * 4.0 / 3.0    # effective radius of earth in meters.
    r = rng * 1000.0                    # distances to gates in meters.

    z = (r ** 2 + Re ** 2 + 2.0 * r * Re * np.sin(theta_e)) ** 0.5 - Re
    z = z + radar_dict['radar_elev']
    s = Re * np.arcsin(r * np.cos(theta_e) / (Re + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    Re = 6371.0 * 1000.0                # radius of earth in meters.
    c = np.sqrt(x*x + y*y) / Re
    phi_0 = radar_lat * np.pi / 180
    azi = np.arctan2(y, x)  # from east to north

    lat = np.arcsin(np.cos(c) * np.sin(phi_0) +
                    np.sin(azi) * np.sin(c) * np.cos(phi_0)) * 180 / np.pi
    lon = (np.arctan2(np.cos(azi) * np.sin(c), np.cos(c) * np.cos(phi_0) -
           np.sin(azi) * np.sin(c) * np.sin(phi_0)) * 180 /
            np.pi + radar_lon)
    lon = np.fmod(lon + 180, 360) - 180
    # height, tmp = np.meshgrid(z, radar_az)
    # s, tmp = np.meshgrid(s, radar_az)
    return lon, lat, z, x, y, s

def ppi_fill(ar):
    out_ar=np.zeros((ar.shape[0]+1,ar.shape[1]),dtype=ar.dtype)
    out_ar[:-1,:]=ar
    out_ar[-1,:]=ar[0,:]
    return out_ar
# rk=rkcfile('/Users/tmin/spectral/PX-1000/20210817/PX-20210817-013642.460.rkc')
# rk=rkcfile('/Users/tmin/spectral/PX-1000/20211111/PX-20211111-004514.730.rkc')
# pulses = rk.getpulses()