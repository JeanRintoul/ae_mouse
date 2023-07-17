/**
 * Transient stream 2 implements a timed generator beginning of a longer streamed measurement. 
 *
 * This example performs a stream mode measurement and writes the data to OscilloscopeStream.csv.
 *
 * Find more information on http://www.tiepie.com/LibTiePie .
 */

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#ifdef __WINNT__
  #include <windows.h>
#else
  #include <poll.h>
  #include <sys/eventfd.h>
#endif
#include <string.h>

#include <libtiepie.h>
#include "CheckStatus.h"
#include "PrintInfo.h"
#include "Utils.h"
#include "npy.h"

#define MAX_SIZE 32
#define EVENT_COUNT 3
#define ndim 3
#define PI 3.141592653589793
#define MAX 20

#define DEVICE_GONE \
  { \
    fprintf(stderr, "Device gone!\n"); \
    status = EXIT_FAILURE; \
    goto exit; \
  }

#define DATA_READY \
  { \
    uint64_t samplesRead = ScpGetData(scp, channelData, channel_count, 0, record_length); \
    for (int j = 0; j < channel_count; j++)  \
      fwrite(channelData[j], sizeof(float), record_length, f);  \
    currentSample += samplesRead;  \
    if(LibGetLastStatus() < LIBTIEPIESTATUS_SUCCESS) \
    { \
      fprintf(stderr, "%s failed: %s\n", raw ? "ScpGetDataRaw" : "ScpGetData", LibGetLastStatusStr()); \
      status = EXIT_FAILURE; \
      goto exit; \
    } \
    n++; \
    printf("\r%0.1f %%", 100.0 * n / num_blocks); \
    fflush(stdout); \
  }

#define DATA_OVERFLOW \
  { \
    fprintf(stderr, "Data overflow!\n"); \
    status = EXIT_FAILURE; \
    goto exit; \
  }

int main(int argc, char* argv[])
{
  int status = EXIT_SUCCESS;
  
  int opt = 0;
  char filename_prefix[MAX];
  // constants which can either be parsed in or set here and recompiled. 
  double sample_frequency             = 5e6;    // 5Mhz sample rate 
  double duration                     = 0.1;    // measured in seconds
  unsigned int resolution             = 0;
  uint64_t record_length              = 50000; // 50 kS
  unsigned int active_channel_count   = 0;
  double current_amplitude            = 4.0; //0.5616; 
  double pressure_amplitude           = 2.0;
  double current_signal_frequency     = 388000;
  double pressure_signal_frequency    = 500000;  

  // while ((opt = getopt(argc, argv, "f:p:")) != -1) {
  while ((opt = getopt(argc, argv, "f:p:d:s:")) != -1) {    
      switch(opt) {
      case 'f':
        sscanf(optarg,"%lf\n",&current_signal_frequency); 
        snprintf(filename_prefix, MAX, "%0.2f", (float)current_signal_frequency) ;
        printf("\ncurrent_signal_frequency(Hz): %s\n",optarg);
      break;
      case 'p':
        printf("\nposition: %s\n",optarg);
        sscanf(optarg,"%s\n",filename_prefix); 
      break;
      case 's':  // sample rate
        sscanf(optarg,"%lf\n",&sample_frequency);
        printf("\nsample_frequency: %s\n",optarg);
      break;   
      case 'd':  // duration
        sscanf(optarg,"%lf\n",&duration);
        printf("\nduration(s): %s\n",optarg);
      break;    
      case '?':
      /* Case when user enters the command as
       * $ ./cmd_exe -i
       */
      if (optopt == 'f') {
      printf("\nMissing mandatory input option");
       // Case when user enters the command as
       // * # ./cmd_exe -o
       
    } else if (optopt == 'p') {
       printf("\nMissing mandatory output option");
    } else {
       printf("\nInvalid option received");
    }
    break;
   }
   }
   // and command line parsing. 

  // Initialize library:
  LibInit();

  // Print library information:
  // printLibraryInfo();

  // Enable network search:
  NetSetAutoDetectEnabled(BOOL8_TRUE);
  CHECK_LAST_STATUS();

  // Update device list:
  LstUpdate();
  CHECK_LAST_STATUS();

  // Try to open an oscilloscope with with stream measurement support:
  LibTiePieHandle_t scp          = LIBTIEPIE_HANDLE_INVALID;
  LibTiePieHandle_t gen_current  = LIBTIEPIE_HANDLE_INVALID;
  LibTiePieHandle_t gen_pressure = LIBTIEPIE_HANDLE_INVALID;

  for(uint32_t index = 0; index < LstGetCount(); index++)
  {

    uint32_t product_id = LstDevGetProductId(IDKIND_INDEX,index);
    // printf("product id: %d\n", product_id);

    if(LstDevCanOpen(IDKIND_INDEX, index, DEVICETYPE_OSCILLOSCOPE))
    {
      scp = LstOpenOscilloscope(IDKIND_INDEX, index);
      CHECK_LAST_STATUS();

      // Check for valid handle and stream measurement support:
      if(scp != LIBTIEPIE_HANDLE_INVALID && (ScpGetMeasureModes(scp) & MM_STREAM))
      {
        printf("Streaming oscilloscope opened: %d\n",product_id);
      }
      else
      {
        scp = LIBTIEPIE_HANDLE_INVALID;
        status = EXIT_FAILURE;
        goto exit_no_mem;
      }
    }

    if(LstDevCanOpen(IDKIND_PRODUCTID,35,DEVICETYPE_GENERATOR)&& product_id==35)
    {
      gen_current = LstOpenGenerator(IDKIND_INDEX, index);
      printf("Generator current opened: %d\n",product_id);
    }
    if(LstDevCanOpen(IDKIND_PRODUCTID,22,DEVICETYPE_GENERATOR) && product_id==22)
    {
      gen_pressure = LstOpenGenerator(IDKIND_INDEX, index);
      printf("Generator pressure opened: %d\n",product_id);
    }

  }

  printf("scp/genc/genp: %d,%d,%d\n",scp,gen_current,gen_pressure);
  if(scp == LIBTIEPIE_HANDLE_INVALID)
  {
    fprintf(stderr, "LstOpenOscilloscope failed: %s\n", LibGetLastStatusStr());
    status = EXIT_FAILURE;
    goto exit_no_mem;
  }

  // Set up the trigger system for the generators. 
  // This works by ....  
  // connect D-SUB pins 3 (EXT3) of both instruments to each other
  // connect D-SUB pins 6 (GND) of both instruments to each other
  // 1. First take a dummy measurement to ensure the clocks on each of the instruments are synced. 
  // 2. Enable trigger output of pressure generator on EXT 3. 
  // 3. Enable EXT 3 trigger source input for both generators and set it to falling edge.  
  // 4. Start both generators, they now wait for trigger. Start scope to measure. 
  // 5. 
  // 
  // First do dummy measurement to time synce both generator clocks. 
  ScpStart(scp);
  CHECK_LAST_STATUS();
  // Wait for measurement to complete:
  while(!ScpIsDataReady(scp) && !ObjIsRemoved(scp))
  {
    sleepMiliSeconds(10); // 10 ms delay, to save CPU time.
  }

  if(ObjIsRemoved(scp))
  {
    fprintf(stderr, "Device gone!");
    status = EXIT_FAILURE;
  }

  // Get the trigger output. 
  const uint16_t trigger_output = DevTrGetOutputIndexById(gen_pressure,TIID_EXT3);
  // Enable the trigger output. 
  DevTrOutSetEnabled(gen_pressure, trigger_output, BOOL8_TRUE);
  // Set the event type to manual. Device, Output, Event.
  DevTrOutSetEvent(gen_pressure,trigger_output,TOE_MANUAL);
  // Output invert. 
  GenSetOutputInvert(gen_pressure, BOOL8_TRUE);

  // Set the trigger inputs
  const uint16_t input_pressure = DevTrGetInputIndexById(gen_pressure, TIID_EXT3);
  // Enable trigger input:
  DevTrInSetEnabled(gen_pressure, input_pressure, BOOL8_TRUE);
  CHECK_LAST_STATUS();
  DevTrInSetKind(gen_pressure, input_pressure, TK_FALLINGEDGE);
  CHECK_LAST_STATUS();

  const uint16_t input_current = DevTrGetInputIndexById(gen_current, TIID_EXT3);
  // // Enable trigger input:
  DevTrInSetEnabled(gen_current, input_current, BOOL8_TRUE);
  CHECK_LAST_STATUS();
  DevTrInSetKind(gen_current, input_current, TK_FALLINGEDGE);
  CHECK_LAST_STATUS();  

  // Channel identities: 
  // 1. rf amplifier output
  // 2. hydrophone probe
  // 3. sr560 dipole probe
  // 4. tiepie voltage output waveform
  // 5. current monitor 
  // 6. voltage monitor
  // 7. probe current monitor
  // 8. probe voltage monitor

  // Define the constants used. 

  bool raw                         = false;
  // Create array of data ranges. 
  // Interestingly, changing these data ranges, seems to be vastly changing the AE amplitude? 
  double dRanges[8] = {20.0,0.2,20.0,20.0,20.0,8.0,20.0,20.0};
  // create array of AC or DC coupling. 
  uint64_t Couplings[8] = {CK_DCV,CK_DCV,CK_DCV,CK_DCV,CK_DCV,CK_DCV,CK_DCV,CK_DCV};
  // uint64_t Couplings[8] = {CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV};
  const uint16_t channel_count = ScpGetChannelCount(scp);

  // Set signal type:
  GenSetSignalType(gen_current, ST_ARBITRARY);
  CHECK_LAST_STATUS();
  GenSetFrequencyMode(gen_current,FM_SAMPLEFREQUENCY );
  GenSetFrequency(gen_current, sample_frequency); // 1 kHz
  CHECK_LAST_STATUS();
  // Set amplitude:
  GenSetAmplitude(gen_current, current_amplitude); // 2 V
  CHECK_LAST_STATUS();
  // Set offset:
  GenSetOffset(gen_current, 0); // 0 V
  CHECK_LAST_STATUS();

  int duration_in_samples = (int)sample_frequency*duration;
  float* data = malloc(sizeof(float) * duration_in_samples);
  float* datap = malloc(sizeof(float) * duration_in_samples);  

  // smaples per wavelength
  double pulse_length = sample_frequency/pressure_signal_frequency;

  int start_time = 0.0; //0.01*duration_in_samples;  // start index  // if total length is 0.1 seconds. and lag thing is 0.07
  int end_time   = pulse_length*15; //0.1*duration_in_samples;  // end index
  //   

  for(int i=0; i<duration_in_samples ; i++)  
  {  
    if (i < (int)start_time) {
      data[i]  = 0.0;
      datap[i] = 0.0;
       // printf("\n %d = %lf", i, data[i]);
    }
    else if (i >=  (int)start_time  && i <= end_time  )
    {
      datap[i] = sinf(2*PI* pressure_signal_frequency * (float)i/sample_frequency);
      data[i]  = sinf(2*PI* current_signal_frequency * (float)i/sample_frequency);  // add a DC ofset on purpose, to see what happens over time. 
    }
    else {
      data[i]  = 0.0; //sinf(2*PI* signal_f1 * (float)i/sample_frequency);
      datap[i] = 0.0;
    }
  }
  printf("\n");  
  // Write .npy file
  // const char* filename = "raw_sig_block.npy";
  // FILE* f = fopen(filename, "wb");
  // skip_npy_header(f);
  // fwrite(data,sizeof(float),duration_in_samples,f);
  // // Close File: 
  // const int array_shape[1] = {duration_in_samples};
  // write_npy_header(f, 1, array_shape, NPY_FLOAT);
  // fclose(f);

  // Load the signal array into the generator:
  GenSetData(gen_current, data, duration_in_samples);
  CHECK_LAST_STATUS();

  if ( GenIsControllable(gen_current)) {
    uint64_t dlength = GenGetDataLength(gen_current);
    CHECK_LAST_STATUS();
    printf("data in generator: %d\n",dlength);
  }
  else {
    printf("generator not controllable\n");
  }

  // Print generator info:
  // printDeviceInfo(gen_current);
  // Enable output:
  GenSetOutputOn(gen_current, BOOL8_TRUE);
  CHECK_LAST_STATUS();


  GenSetSignalType(gen_pressure, ST_ARBITRARY);
  CHECK_LAST_STATUS();
  GenSetFrequencyMode(gen_pressure,FM_SAMPLEFREQUENCY );
  GenSetFrequency(gen_pressure, sample_frequency); // 1 kHz
  CHECK_LAST_STATUS();

  // Set amplitude:
  GenSetAmplitude(gen_pressure, pressure_amplitude); // 2 V
  CHECK_LAST_STATUS();

  // Set offset:
  GenSetOffset(gen_pressure, 0); // 0 V
  CHECK_LAST_STATUS();

  // Enable output:
  GenSetOutputOn(gen_pressure, BOOL8_TRUE);
  CHECK_LAST_STATUS();

  // Load the signal array into the generator:
  GenSetData(gen_pressure, datap, duration_in_samples);
  CHECK_LAST_STATUS();

  // Start signal generation:
  GenStart(gen_current);
  CHECK_LAST_STATUS();

  // printf("last status: %d\n",CHECK_LAST_STATUS());
  GenStart(gen_pressure);
  CHECK_LAST_STATUS();

  // set up a trigger output, and just start it immediately... since I have a time gap, hopefully it'd just start at the same time, but a little way into the data.  
  DevTrOutTrigger(gen_pressure, trigger_output);
  CHECK_LAST_STATUS();  

  //  Once the scp is set to stream, the generators are not controllable. 
  // Set measure mode:
  ScpSetMeasureMode(scp, MM_STREAM);

  if(active_channel_count == 0 || active_channel_count > channel_count)
    active_channel_count = channel_count;
  printf("Active channel count: %u\n", active_channel_count);
  for(uint16_t i = 0; i < channel_count; i++)
    printf("  Ch%u: %s\n", i + 1, ScpChSetEnabled(scp, i, i < active_channel_count ? BOOL8_TRUE : BOOL8_FALSE) ? "enabled" : "disabled");

  if(resolution != 0)
  {
    ScpSetResolution(scp, resolution);
  }

  sample_frequency = ScpSetSampleFrequency(scp, sample_frequency);
  printf("Sample frequency: %f MHz\n", sample_frequency / 1e6);

  printf("Resolution: %u bit\n", ScpGetResolution(scp));

  record_length = ScpSetRecordLength(scp, record_length);
  printf("Record length: %" PRIu64 " Samples\n", record_length);

  printf("Data rate: %f MB/s\n", (active_channel_count * ceil(ScpGetResolution(scp) / 8.0) * sample_frequency) / 1e6);

  unsigned int num_blocks = ceil(duration * sample_frequency / record_length);
  duration = (record_length * num_blocks) / sample_frequency;
  printf("Duration: %f s\n", duration);

  printf("Data type: %s\n", raw ? "raw" : "float");

  // printf("channel count %d ",channelCount);
  // For all channels:
  for(uint16_t ch = 0; ch < (channel_count); ch++)
  {
    // Enable channel to measure it:
    ScpChSetEnabled(scp, ch, BOOL8_TRUE);
    CHECK_LAST_STATUS();
    // Set coupling:
    ScpChSetCoupling(scp, ch, Couplings[ch]); // DC Volt
    CHECK_LAST_STATUS();
    // Set range:
    ScpChSetRange(scp, ch, dRanges[ch]); // 8 V
    CHECK_LAST_STATUS();

    double range = ScpChGetRange(scp,ch);
    if (range != dRanges[ch])
    {
       printf("Note: Range modified ch %d = measured: %f desired: %f\n",ch,range, dRanges[ch]);
    }
    // uint64_t coupling = ScpChGetCoupling(scp,ch);
    // printf("Coupling ch %" PRIu64 "\n",coupling);
  }

  // Print generator info:
  // printDeviceInfo(gen_pressure);
  // Print oscilloscope info:
  // printDeviceInfo(scp);

  {

  // Create data buffers:
  float** channelData = malloc(sizeof(float*) * channel_count);
  for(uint16_t ch = 0; ch < channel_count; ch++)
  {
    channelData[ch] = malloc(sizeof(float) * record_length);
  }

// Setup events:
#ifdef __WINNT__
    HANDLE events[EVENT_COUNT] = {
      CreateEventA(NULL, FALSE, FALSE, NULL),
      CreateEventA(NULL, FALSE, FALSE, NULL),
      CreateEventA(NULL, FALSE, FALSE, NULL)
    };

    DevSetEventRemoved(scp, events[0]);
    ScpSetEventDataReady(scp, events[1]);
    ScpSetEventDataOverflow(scp, events[2]);
#else
    int fd_removed = eventfd(0, EFD_NONBLOCK);
    int fd_data_ready = eventfd(0, EFD_NONBLOCK);
    int fd_data_overflow = eventfd(0, EFD_NONBLOCK);

    DevSetEventRemoved(scp, fd_removed);
    ScpSetEventDataReady(scp, fd_data_ready);
    ScpSetEventDataOverflow(scp, fd_data_overflow);

    struct pollfd fds[EVENT_COUNT] = {
      {fd: fd_removed, events: POLLIN, revents: 0},
      {fd: fd_data_ready, events: POLLIN, revents: 0},
      {fd: fd_data_overflow, events: POLLIN, revents: 0}
    };
#endif
    // 
    // write .npy file that starts
    char file_ending[20] = "_stream.npy";
    char file_label[100];
    strcpy(file_label,file_ending);
    strcat(filename_prefix,file_label);
    printf("\nfilename result: %s\n",filename_prefix);
    FILE* f = fopen(filename_prefix, "wb");
    skip_npy_header(f);

    // Write .npy file
    // const char* filename = "transient_stream_data.npy";
    // FILE* f = fopen(filename, "wb");
    // skip_npy_header(f);

    uint64_t currentSample = 0;

    if(!ScpStart(scp))
    {
      fprintf(stderr, "ScpStart failed: %s\n", LibGetLastStatusStr());
      status = EXIT_FAILURE;
      goto exit;
    }

    unsigned int n = 0;
#ifdef __WINNT__
    DWORD r;
    while(n < num_blocks && (r = WaitForMultipleObjects(EVENT_COUNT, events, FALSE, INFINITE)) >= WAIT_OBJECT_0)
    {
      if(r == WAIT_OBJECT_0)
      {
        DEVICE_GONE
      }
      else if(r == WAIT_OBJECT_0 + 1)
      {
        DATA_READY
      }
      else if(r == WAIT_OBJECT_0 + 2)
      {
        DATA_OVERFLOW
      }
      else
      {
        fprintf(stderr, "WaitForMultipleObjects() returned: %lu\n", r);
        status = EXIT_FAILURE;
        goto exit;
      }
    }
#else
    int r;
    while(n < num_blocks && (r = poll(fds, EVENT_COUNT, -1)) >= 0)
    {
      for(int i = 0; i < EVENT_COUNT; i++)
      {
        if(fds[i].revents & POLLIN)
        {
          eventfd_t tmp;
          eventfd_read(fds[i].fd, &tmp);

          if(fds[i].fd == fd_removed)
          {
            DEVICE_GONE
          }

          if(fds[i].fd == fd_data_overflow)
          {
            DATA_OVERFLOW
          }

          if(fds[i].fd == fd_data_ready)
          {
            DATA_READY
          }
        }

        fds[i].revents = 0;
      }
    }
#endif

    // Close File: 
    const int array_shape[ndim] = {num_blocks,channel_count,record_length};
    write_npy_header(f, ndim, array_shape, NPY_FLOAT);
    fclose(f);

    // Stop measurement:
    ScpStop(scp);

    // Stop generator:
    GenStop(gen_current);
    CHECK_LAST_STATUS();
    GenStop(gen_pressure);
    CHECK_LAST_STATUS();

    // Disable output:
    GenSetOutputOn(gen_current, BOOL8_FALSE);
    CHECK_LAST_STATUS();
    GenSetOutputOn(gen_pressure, BOOL8_FALSE);
    CHECK_LAST_STATUS();

    // Close oscilloscope:
    ObjClose(scp);
    CHECK_LAST_STATUS();

    // Close generator:
    ObjClose(gen_current);
    CHECK_LAST_STATUS();

    // Close generator:
    ObjClose(gen_pressure);
    CHECK_LAST_STATUS();

exit:
    // Delete data buffers:
    for(uint16_t ch = 0; ch < channel_count; ch++)
      free(channelData[ch]);
    free(channelData);

  }

exit_no_mem:
  LibExit();
  return status;
}






