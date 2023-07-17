/**
 * OscilloscopeStream.c
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

#define PI 3.141592653589793
#define ndim 2
#define MAX 20

int main(int argc, char* argv[])
{
  int status = EXIT_SUCCESS;

  char position[MAX];
  if(argc==1)
      printf("\nNo Extra Command Line Argument Passed Other Than Program Name");
  if(argc>=2)
  {
    // printf("\nNumber Of Arguments Passed: %d",argc);
    for(int counter=0;counter<argc;counter++)
    {
      // printf("\nargv[%d]: %s\n",counter,argv[counter]);
      if(counter ==2)
      { 
        printf("\nposition: %s\n",argv[2]);
        sscanf(argv[2],"%s\n",position); 
      }      
    }
  }
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
  // LibTiePieHandle_t gen_current  = LIBTIEPIE_HANDLE_INVALID;
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
      if(scp != LIBTIEPIE_HANDLE_INVALID && (ScpGetMeasureModes(scp) & MM_BLOCK))
      {
        printf("Block oscilloscope opened: %d\n",product_id);
      }
      else
      {
        scp = LIBTIEPIE_HANDLE_INVALID;
        status = EXIT_FAILURE;
      }
    }

    // if(LstDevCanOpen(IDKIND_PRODUCTID,35,DEVICETYPE_GENERATOR)&& product_id==35)
    // {
    //   gen_current = LstOpenGenerator(IDKIND_INDEX, index);
    //   printf("Generator current opened: %d\n",product_id);
    // }
    if(LstDevCanOpen(IDKIND_PRODUCTID,22,DEVICETYPE_GENERATOR) && product_id==22)
    {
      gen_pressure = LstOpenGenerator(IDKIND_INDEX, index);
      printf("Generator pressure opened: %d\n",product_id);
    }

  }

  // printf("scp/genc/genp: %d,%d,%d\n",scp,gen_current,gen_pressure);
  if(scp == LIBTIEPIE_HANDLE_INVALID)
  {
    fprintf(stderr, "LstOpenOscilloscope failed: %s\n", LibGetLastStatusStr());
    status = EXIT_FAILURE;
  }
  // Where am I measuring the electric field? Differential channel? 
  // Let's put thecable on the last channnel


  // Channel identities: 
  // 1. rf amplifier output
  // 2. hydrophone probe
  // 
  // chan last 8 is measuring the efield from the dipole. 
  // 
  // Define the constants used. 
  double sample_frequency   = 1e8;      // 5Mhz sample rate 
  double duration           = 0.008;    // measured in seconds
  unsigned int resolution   = 0;
  uint64_t record_length    = duration*sample_frequency; // N
  unsigned int active_channel_count = 0;
  // printf("data sample length N: %d\n",N);
  // double current_amplitude         = 10.0;  // V 
  // double current_signal_frequency  = 1000;    
  double pressure_amplitude        = 0.031; // focused transducer
  // double pressure_amplitude        = 2.0;    // planar transducer
  double pressure_signal_frequency = 5e5;
  //double pressure_signal_frequency = 672800;  
  bool raw = false;
  // Create array of data ranges. 
  double dRanges[8] = {80.0,0.4,4.0,4.0,4.0,2.0,0.2,0.4};
  // Create array of enabled channels. 
  bool chan_enabled[8] = {true,true,true,true,true,true,true,true};
  // Create array of AC or DC coupling. 
  uint64_t Couplings[8] = {CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV,CK_ACV};
  // 
  const uint16_t channel_count = ScpGetChannelCount(scp);
  // 
  // Set up the generators 
  // Set signal type:
  // GenSetSignalType(gen_current, ST_ARBITRARY);
  // CHECK_LAST_STATUS();

  // // Select frequency mode:
  // GenSetFrequencyMode(gen_current, FM_SAMPLEFREQUENCY);
  // GenSetFrequency(gen_current, sample_frequency); // set the sample frequency. 
  // CHECK_LAST_STATUS();

  // start sine waveform. 
  // Set signal type:
  GenSetSignalType(gen_pressure, ST_SINE);
  CHECK_LAST_STATUS();

  // Set frequency:
  GenSetFrequency(gen_pressure, pressure_signal_frequency); // 499.99 kHz
  CHECK_LAST_STATUS();

  // end typical sinewaveform

  // Set amplitude:
  GenSetAmplitude(gen_pressure, pressure_amplitude); // 2 V
  CHECK_LAST_STATUS();

  // Set offset:
  GenSetOffset(gen_pressure, 0); // 0 V
  CHECK_LAST_STATUS();

  // Enable output:
  GenSetOutputOn(gen_pressure, BOOL8_TRUE);
  CHECK_LAST_STATUS();

  // Create signal array:
  // float data[record_length];
  // double f = 8000; 
  // // float gain = 0.5;
  // // sin(2*pi*f/fs)
  // for(unsigned int i = 0; i < record_length; i++)
  // { 
  //   // Now I want to move between 0 and 12V in amplitude. 
  //   data[i] = i*sinf(((float) i * PI * 2 * f) / (float) (sample_frequency));
  // }

  // // Load the signal array into the generator:
  // GenSetData(gen_current, data, record_length);
  // CHECK_LAST_STATUS();

  // Print generator info:
  // printDeviceInfo(gen_current);

  // Set signal frequency:
  // GenSetFrequencyMode(gen_pressure,FM_SIGNALFREQUENCY);    
  // GenSetFrequency(gen_pressure, pressure_signal_frequency); // 1 kHz
  // CHECK_LAST_STATUS();

  // // Set signal type:
  // GenSetSignalType(gen_pressure, ST_SINE);
  // CHECK_LAST_STATUS();

  // // Set amplitude:
  // GenSetAmplitude(gen_pressure, pressure_amplitude); // 2 V
  // CHECK_LAST_STATUS();

  // // Set offset:
  // GenSetOffset(gen_pressure, 0); // 0 V
  // CHECK_LAST_STATUS();

  // // Enable output:
  // GenSetOutputOn(gen_pressure, BOOL8_TRUE);
  // CHECK_LAST_STATUS();


  // GenStart(gen_pressure);
  // CHECK_LAST_STATUS();

  // Set up the oscilloscopes: 
  // Set measure mode:
  ScpSetMeasureMode(scp, MM_BLOCK);

  if(active_channel_count == 0 || active_channel_count > channel_count)
    active_channel_count = channel_count;
  printf("Active channel count: %u\n", active_channel_count);
  for(uint16_t i = 0; i < channel_count; i++)
    printf("  Ch%u: %s\n", i + 1, ScpChSetEnabled(scp, i, i < active_channel_count ? BOOL8_TRUE : BOOL8_FALSE) ? "enabled" : "disabled");

  if(resolution != 0)
  {
    ScpSetResolution(scp, resolution);
  }

  // set presample ratio 
  // ScpSetPreSampleRatio(scp,0.05);
  ScpSetPreSampleRatio(scp,0.0);


  sample_frequency = ScpSetSampleFrequency(scp, sample_frequency);
  printf("Sample frequency: %f MHz\n", sample_frequency / 1e6);

  printf("Resolution: %u bit\n", ScpGetResolution(scp));
  record_length = ScpSetRecordLength(scp, record_length);
  printf("Record length: %" PRIu64 " Samples\n", record_length);

  printf("Data rate: %f MB/s\n", (active_channel_count * ceil(ScpGetResolution(scp) / 8.0) * sample_frequency) / 1e6);

  unsigned int num_blocks = ceil(duration * sample_frequency / record_length);
  printf("Num blocks: %u \n", num_blocks);

  duration = (record_length * num_blocks) / sample_frequency;
  printf("Duration: %f s\n", duration);

  printf("Data type: %s\n", raw ? "raw" : "float");

  // For all channels:
  for(uint16_t ch = 0; ch < (channel_count); ch++)
  {
    // Enable channel to measure it:
    // ScpChSetEnabled(scp, ch, BOOL8_TRUE);
    ScpChSetEnabled(scp, ch, chan_enabled[ch]);
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

  // Set trigger timeout:
  ScpSetTriggerTimeOut(scp, 1); // 1 s
  CHECK_LAST_STATUS();

  // Disable all channel trigger sources:
  for(uint16_t ch = 0; ch < channel_count; ch++)
  {
    ScpChTrSetEnabled(scp, ch, BOOL8_FALSE);
    CHECK_LAST_STATUS();
  }

  // Locate trigger input:
  // const uint16_t index = DevTrGetInputIndexById(scp, TIID_GENERATOR_NEW_PERIOD); // or TIID_GENERATOR_START || TIID_GENERATOR_STOP
  // const uint16_t index = DevTrGetInputIndexById(scp, TIID_GENERATOR_START); // or TIID_GENERATOR_START || TIID_GENERATOR_STOP
  // CHECK_LAST_STATUS();

  // # To use GENERATOR_START of the second HS5 (Ch3 and Ch4):
  const uint16_t index = DevTrGetInputIndexById(scp, (DN_SUB_SECOND << TIOID_SHIFT_DN) | TIID_GENERATOR_START);
  CHECK_LAST_STATUS();

  if(index != LIBTIEPIE_TRIGGERIO_INDEX_INVALID)
  {
    // Enable trigger input:
    DevTrInSetEnabled(scp, index, BOOL8_TRUE);
    CHECK_LAST_STATUS();
  }


  if (ScpHasTriggerHoldOff(scp))
  {
     ScpSetTriggerHoldOffCount(scp,TH_ALLPRESAMPLES);
  }

  // Block measurement code: 
  // Start measurement:
  ScpStart(scp);
  CHECK_LAST_STATUS();

  // Start signal generation:
  GenStart(gen_pressure);
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
  else if(ScpIsDataReady(scp))
  {
    // Create data buffers:
    float** channelData = malloc(sizeof(float*) * channel_count);
    for(uint16_t ch = 0; ch < channel_count; ch++)
    {
      channelData[ch] = malloc(sizeof(float) * record_length);
    }

    // Get data:

    // Get the data from the scope:
    record_length = ScpGetData(scp, channelData, channel_count, 0, record_length);
    // record_length = ScpGetData(scp, channelData, channel_count, Start , ValidSamples);
    CHECK_LAST_STATUS();

    char file_ending[20] = "_pressure_block.npy";
    char file_label[100];
    strcpy(file_label,file_ending);
    strcat(position,file_label);
    printf("\nfilename result: %s\n",position);

    // Write .npy file
    // const char* filename = "current_check_block.npy";
    FILE* f = fopen(position, "wb");
    skip_npy_header(f);

    // Write to file.  
    for (int j = 0; j < channel_count; j++)  
      fwrite(channelData[j], sizeof(float), record_length, f);  

    // Write the data to csv:
    // for(uint64_t i = 0; i < record_length; i++)
    // {
    //   for(uint16_t ch = 0; ch < channel_count; ch++)
    //   {
    //     if (ch == 2 && i >3500)
    //     {
    //       printf(";%f", channelData[ch][i]);
    //     }

    //   }
    // }


    // for (int j = 0; j < record_length; j++)  
    //   fwrite(channelData[j], sizeof(float), channel_count, f);  

    // Close File: 
    const int array_shape[ndim] = {channel_count,record_length};
    write_npy_header(f, ndim, array_shape, NPY_FLOAT);
    fclose(f);

    // Free data buffers:
    for(uint16_t ch = 0; ch < channel_count; ch++)
    {
      free(channelData[ch]);
    }
    free(channelData);
  }

    // Stop generator:
    GenStop(gen_pressure);
    CHECK_LAST_STATUS();
    // GenStop(gen_pressure);
    // CHECK_LAST_STATUS();

    // Disable output:
    GenSetOutputOn(gen_pressure, BOOL8_FALSE);
    CHECK_LAST_STATUS();
    // GenSetOutputOn(gen_pressure, BOOL8_FALSE);
    // CHECK_LAST_STATUS();

    // Close oscilloscope:
    ObjClose(scp);
    CHECK_LAST_STATUS();

    // Close generator:
    ObjClose(gen_pressure);
    CHECK_LAST_STATUS();

    // Close generator:
    // ObjClose(gen_pressure);
    // CHECK_LAST_STATUS();

  LibExit();
  return status;
}






