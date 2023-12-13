/**
 * PrintInfo.c
 *
 * This file is part of the LibTiePie programming examples.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#include "PrintInfo.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

const char* GeneratorModes[TIEPIE_HW_GMN_COUNT] = {"Continuous", "Burst count", "Gated periods", "Gated", "Gated period start", "Gated period finish", "Gated run", "Gated run output", "Burst sample count", "Burst sample count output", "Burst segment count", "Burst segment count output"};
const char* ClockOutputTypes[TIEPIE_HW_CON_COUNT] = {"Disabled", "Sample", "Fixed"};
const char* ClockSources[TIEPIE_HW_CSN_COUNT] = {"External", "Internal"};
const char* ConnectorTypes[TIEPIE_HW_CONNECTORTYPE_COUNT] = {"BNC", "Banana", "Power plug"};
const char* Couplings[TIEPIE_HW_CKN_COUNT] = {"DCV", "ACV", "DCA", "ACA", "Ohm"};
const char* DeviceTypes[TIEPIE_HW_DEVICETYPE_COUNT] = {"Oscilloscope", "Generator"};
const char* FrequencyModes[TIEPIE_HW_FMN_COUNT] = {"Signal frequency", "Sample frequency"};
const char* MeasureModes[TIEPIE_HW_MMN_COUNT] = {"Stream", "Block"};
const char* AutoResolutionModes[TIEPIE_HW_ARMN_COUNT] = {"Disabled", "Native only", "All"};
const char* SignalTypes[TIEPIE_HW_STN_COUNT] = {"Sine", "Triangle", "Square", "DC", "Noise", "Arbitrary", "Pulse"};
const char* TriggerConditions[TIEPIE_HW_TCN_COUNT] = {"None", "Smaller", "Larger", "Inside", "Outside"};
const char* TriggerKinds[TIEPIE_HW_TKN_COUNT] = {"Rising edge", "Falling edge", "In window", "Out Window", "Any edge", "Enter window", "Exit window", "Pulse width positive", "Pulse width negative", "Pulse width either", "Runt pulse positive", "Runt pulse negative", "Runt pulse either", "Interval rising", "Interval falling"};
const char* TriggerLevelModes[TIEPIE_HW_TLMN_COUNT] = {"Relative", "Absolute"};
const char* TriggerOutputEvents[TIEPIE_HW_TOEN_COUNT] = {"Generator start", "Generator stop", "Generator new period", "Oscilloscope running", "Oscilloscope triggered", "Manual"};
const char* ServerErrorCodes[] = {"None", "Unknown", "Connection refused", "Network unreachable", "Timed out", "Hostname lookup failed"};
const char* ServerStatuses[] = {"Disconnected", "Connecting", "Connected", "Disconnecting"};

void printLibraryInfo()
{
  printf("Library:\n");

  // Print library version:
  printf("  Version: ");
  printVersion(tiepie_hw_get_version());

  // Print library configuration:
  uint32_t length = tiepie_hw_get_config(NULL, 0);
  uint8_t configuration[sizeof(uint8_t) * length];
  length = tiepie_hw_get_config(configuration, length);

  printf("  Configuration: 0x");
  for(uint32_t i = 0; i < length; i++)
    printf("%02" PRIx8, configuration[i]);
  printf("\n");
}

void printDeviceInfo(tiepie_hw_handle dev)
{
  if(dev == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid device handle in printDeviceInfo()\n");
    return;
  }

  printf("Device:\n");

  // Name:
  uint32_t length = tiepie_hw_device_get_name(dev, NULL, 0) + 1;
  char deviceName[length];
  length = tiepie_hw_device_get_name(dev, deviceName, length);
  printf("  Name                      : %s\n", deviceName);

  // Name short:
  length = tiepie_hw_device_get_name_short(dev, NULL, 0) + 1;
  char deviceNameShort[length];
  length = tiepie_hw_device_get_name_short(dev, deviceNameShort, length);
  printf("  Short name                : %s\n", deviceNameShort);

  printf("  Serial number             : %" PRIu32 "\n", tiepie_hw_device_get_serial_number(dev));

  printf("  Calibration data          : ");
  printDate(tiepie_hw_device_get_calibration_date(dev));
  printf("\n");

  printf("  Product id                : %" PRIu32 "\n", tiepie_hw_device_get_product_id(dev));

  length = tiepie_hw_device_get_ip_address(dev, NULL, 0) + 1;
  char deviceIPaddress[length];
  length = tiepie_hw_device_get_ip_address(dev, deviceIPaddress, length);
  if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
  {
    printf("  IPv4 address              : %s\n", deviceIPaddress);
    printf("\n");
  }

  const uint16_t ipPort = tiepie_hw_device_get_ip_port(dev);
  if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
  {
    printf("  IP port                   : %" PRIu16 "\n", ipPort);
  }

  printf("  Has battery               : %s\n", boolToStr(tiepie_hw_device_has_battery(dev)));

  if(tiepie_hw_device_has_battery(dev))
  {
    printf("  Battery:\n");

    const int8_t batteryCharge = tiepie_hw_device_get_battery_charge(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Charge                  : %" PRIi8 " %%\n", batteryCharge);
    }

    const int32_t batteryTimeToEmpty = tiepie_hw_device_get_battery_time_to_empty(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Time to empty           : %" PRIi32 " minutes\n", batteryTimeToEmpty);
    }

    const int32_t batteryTimeToFull = tiepie_hw_device_get_battery_time_to_full(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Time to full            : %" PRIi32 " minutes\n", batteryTimeToFull);
    }

    const uint16_t isBatteryChargerConnected = tiepie_hw_device_is_battery_charger_connected(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Charger connected       : %s\n", boolToStr(isBatteryChargerConnected));
    }

    const uint16_t isBatteryCharging = tiepie_hw_device_is_battery_charging(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Charging                : %s\n", boolToStr(isBatteryCharging));
    }

    const uint16_t isBatteryBroken = tiepie_hw_device_is_battery_broken(dev);
    if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
    {
      printf("    Broken                  : %s\n", boolToStr(isBatteryBroken));
    }
  }

  switch(tiepie_hw_device_get_type(dev))
  {
    case TIEPIE_HW_DEVICETYPE_OSCILLOSCOPE:
      printOscilloscopeInfo(dev);
      break;

    case TIEPIE_HW_DEVICETYPE_GENERATOR:
      printGeneratorInfo(dev);
      break;
  }
}

void printOscilloscopeInfo(tiepie_hw_handle scp)
{
  if(scp == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid device handle in printOscilloscopeInfo()\n");
    return;
  }

  const uint16_t channelCount = tiepie_hw_oscilloscope_get_channel_count(scp);

  printf("Oscilloscope:\n");
  printf("  Channel count             : %" PRIu16 "\n", channelCount);
  printf("  SureConnect               : %s\n", boolToStr(tiepie_hw_oscilloscope_has_sureconnect(scp)));
  printf("  Measure modes             : ");
  printMeasureMode(tiepie_hw_oscilloscope_get_measure_modes(scp));
  printf("\n");
  printf("  Measure mode              : ");
  printMeasureMode(tiepie_hw_oscilloscope_get_measure_mode(scp));
  printf("\n");
  printf("  Auto resolution modes     : ");
  printAutoResolutionMode(tiepie_hw_oscilloscope_get_auto_resolution_modes(scp));
  printf("\n");
  printf("  Auto resolution mode      : ");
  printAutoResolutionMode(tiepie_hw_oscilloscope_get_auto_resolution_mode(scp));
  printf("\n");

  uint8_t resolutionCount = tiepie_hw_oscilloscope_get_resolutions(scp, NULL, 0);
  uint8_t resolutions[sizeof(uint8_t) * resolutionCount];
  resolutionCount = tiepie_hw_oscilloscope_get_resolutions(scp, resolutions, resolutionCount);

  printf("  Resolutions               : ");
  for(uint8_t i = 0; i < resolutionCount; i++)
  {
    if(i != 0)
      printf(", ");

    printf("%" PRIu8, resolutions[i]);
  }
  printf("\n");

  printf("  Resolution                : %" PRIu8 "\n", tiepie_hw_oscilloscope_get_resolution(scp));
  printf("  Resolution enhanced       : %s\n", boolToStr(tiepie_hw_oscilloscope_is_resolution_enhanced(scp)));
  printf("  Clock outputs             : ");
  printClockOutput(tiepie_hw_oscilloscope_get_clock_outputs(scp));
  printf("\n");
  printf("  Clock output              : ");
  printClockOutput(tiepie_hw_oscilloscope_get_clock_output(scp));
  printf("\n");

  uint32_t clockOutputFrequencyCount = tiepie_hw_oscilloscope_get_clock_output_frequencies(scp, NULL, 0);
  if(clockOutputFrequencyCount > 0)
  {
    double clockOutputFrequencies[sizeof(double) * clockOutputFrequencyCount];
    clockOutputFrequencyCount = tiepie_hw_oscilloscope_get_clock_output_frequencies(scp, clockOutputFrequencies, clockOutputFrequencyCount);

    printf("  Clock output frequencies  : ");
    for(uint32_t i = 0; i < clockOutputFrequencyCount; i++)
    {
      if(i != 0)
        printf(", ");

      printf("%f", clockOutputFrequencies[i]);
    }
    printf("\n");
    printf("  Clock output frequency    : %f\n", tiepie_hw_oscilloscope_get_clock_output_frequency(scp));
  }

  printf("  Clock sources             : ");
  printClockSource(tiepie_hw_oscilloscope_get_clock_sources(scp));
  printf("\n");
  printf("  Clock source              : ");
  printClockSource(tiepie_hw_oscilloscope_get_clock_source(scp));
  printf("\n");

  uint32_t clockSourceFrequencyCount = tiepie_hw_oscilloscope_get_clock_source_frequencies(scp, NULL, 0);
  if(clockSourceFrequencyCount > 0)
  {
    double clockSourceFrequencies[sizeof(double) * clockSourceFrequencyCount];
    clockSourceFrequencyCount = tiepie_hw_oscilloscope_get_clock_source_frequencies(scp, clockSourceFrequencies, clockSourceFrequencyCount);

    printf("  Clock source frequencies  : ");
    for(uint32_t i = 0; i < clockSourceFrequencyCount; i++)
    {
      if(i != 0)
        printf(", ");

      printf("%f", clockSourceFrequencies[i]);
    }
    printf("\n");
    printf("  Clock source frequency    : %f\n", tiepie_hw_oscilloscope_get_clock_source_frequency(scp));
  }

  printf("  Record length max         : %" PRIu64 "\n", tiepie_hw_oscilloscope_get_record_length_max(scp));
  printf("  Record length             : %" PRIu64 "\n", tiepie_hw_oscilloscope_get_record_length(scp));
  printf("  Sample rate max           : %f\n", tiepie_hw_oscilloscope_get_sample_rate_max(scp));
  printf("  Sample rate               : %f\n", tiepie_hw_oscilloscope_get_sample_rate(scp));

  if(tiepie_hw_oscilloscope_get_measure_mode(scp) == TIEPIE_HW_MM_BLOCK)
  {
    printf("  Segment count max         : %" PRIu32 "\n", tiepie_hw_oscilloscope_get_segment_count_max(scp));
    printf("  Segment count             : %" PRIu32 "\n", tiepie_hw_oscilloscope_get_segment_count(scp));
  }

  if(tiepie_hw_oscilloscope_has_trigger(scp))
  {
    printf("  Pre sample ratio          : %f\n", tiepie_hw_oscilloscope_get_pre_sample_ratio(scp));

    double triggerTimeout = tiepie_hw_oscilloscope_trigger_get_timeout(scp);
    printf("  Trigger time out          : ");
    if(triggerTimeout == TIEPIE_HW_TO_INFINITY)
      printf("Infinite\n");
    else
      printf("%f\n", triggerTimeout);

    if(tiepie_hw_oscilloscope_trigger_has_delay(scp))
    {
      printf("  Trigger delay max         : %f\n", tiepie_hw_oscilloscope_trigger_get_delay_max(scp));
      printf("  Trigger delay             : %f\n", tiepie_hw_oscilloscope_trigger_get_delay(scp));
    }

    if(tiepie_hw_oscilloscope_has_presamples_valid(scp))
    {
      printf("  Presamples valid          : %s\n", boolToStr(tiepie_hw_oscilloscope_get_presamples_valid(scp)));
    }
  }

  for(uint16_t ch = 0; ch < channelCount; ch++)
  {
    printf("  Channel%" PRIu16 ":\n", (ch + 1));
    printf("    Connector type          : ");
    printConnectorType(tiepie_hw_oscilloscope_channel_get_connector_type(scp, ch));
    printf("\n");
    printf("    Differential            : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_is_differential(scp, ch)));
    printf("    Impedance               : %f\n", tiepie_hw_oscilloscope_channel_get_impedance(scp, ch));
    printf("    Connection test         : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_has_sureconnect(scp, ch)));
    printf("    Available               : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_is_available(scp, ch)));
    printf("    Enabled                 : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_get_enabled(scp, ch)));

    uint32_t bandwidthCount = tiepie_hw_oscilloscope_channel_get_bandwidths(scp, ch, NULL, 0);
    double bandwidths[sizeof(double) * bandwidthCount];
    bandwidthCount = tiepie_hw_oscilloscope_channel_get_bandwidths(scp, ch, bandwidths, bandwidthCount);

    printf("    Bandwidths              : ");
    for(uint32_t i = 0; i < bandwidthCount; i++)
    {
      if(i != 0)
        printf(", ");

      printf("%f", bandwidths[i]);
    }
    printf("\n");
    printf("    Bandwidth               : %f\n", tiepie_hw_oscilloscope_channel_get_bandwidth(scp, ch));

    printf("    Couplings               : ");
    printCoupling(tiepie_hw_oscilloscope_channel_get_couplings(scp, ch));
    printf("\n");
    printf("    Coupling                : ");
    printCoupling(tiepie_hw_oscilloscope_channel_get_coupling(scp, ch));
    printf("\n");
    printf("    Auto ranging            : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_get_auto_ranging(scp, ch)));

    uint32_t rangeCount = tiepie_hw_oscilloscope_channel_get_ranges(scp, ch, NULL, 0);
    double ranges[sizeof(double) * rangeCount];
    rangeCount = tiepie_hw_oscilloscope_channel_get_ranges(scp, ch, ranges, rangeCount);

    printf("    Ranges                  : ");
    for(uint32_t i = 0; i < rangeCount; i++)
    {
      if(i != 0)
        printf(", ");

      printf("%f", ranges[i]);
    }
    printf("\n");

    printf("    Range                   : %f\n", tiepie_hw_oscilloscope_channel_get_range(scp, ch));
    if(tiepie_hw_oscilloscope_channel_has_safeground(scp, ch))
    {
      printf("    SafeGround enabled      : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_get_safeground_enabled(scp, ch)));
      printf("    SafeGround threshold max: %f\n", tiepie_hw_oscilloscope_channel_get_safeground_threshold_max(scp, ch));
      printf("    SafeGround threshold min: %f\n", tiepie_hw_oscilloscope_channel_get_safeground_threshold_min(scp, ch));
      printf("    SafeGround threshold    : %f\n", tiepie_hw_oscilloscope_channel_get_safeground_threshold(scp, ch));
    }

    if(tiepie_hw_oscilloscope_channel_has_trigger(scp, ch))
    {
      printf("  Trigger:\n");
      printf("    Available               : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_trigger_is_available(scp, ch)));
      printf("    Enabled                 : %s\n", boolToStr(tiepie_hw_oscilloscope_channel_trigger_get_enabled(scp, ch)));
      printf("    Kinds                   : ");
      printTriggerKind(tiepie_hw_oscilloscope_channel_trigger_get_kinds(scp, ch));
      printf("\n");
      printf("    Kind                    : ");
      printTriggerKind(tiepie_hw_oscilloscope_channel_trigger_get_kind(scp, ch));
      printf("\n");
      printf("    Level modes             : ");
      printTriggerLevelMode(tiepie_hw_oscilloscope_channel_trigger_get_level_modes(scp, ch));
      printf("\n");
      printf("    Level mode              : ");
      printTriggerLevelMode(tiepie_hw_oscilloscope_channel_trigger_get_level_mode(scp, ch));
      printf("\n");

      const uint32_t triggerLevelCount = tiepie_hw_oscilloscope_channel_trigger_get_level_count(scp, ch);
      printf("    Levels                  : ");
      for(uint32_t i = 0; i < triggerLevelCount; i++)
      {
        if(i != 0)
          printf(", ");

        printf("%f", tiepie_hw_oscilloscope_channel_trigger_get_level(scp, ch, i));
      }
      printf("\n");

      const uint32_t triggerHysteresisCount = tiepie_hw_oscilloscope_channel_trigger_get_hysteresis_count(scp, ch);
      printf("    Hystereses              : ");
      for(uint32_t i = 0; i < triggerHysteresisCount; i++)
      {
        if(i != 0)
          printf(", ");

        printf("%f", tiepie_hw_oscilloscope_channel_trigger_get_hysteresis(scp, ch, i));
      }
      printf("\n");

      printf("    Conditions              : ");
      printTriggerCondition(tiepie_hw_oscilloscope_channel_trigger_get_conditions(scp, ch));
      printf("\n");
      if(tiepie_hw_oscilloscope_channel_trigger_get_conditions(scp, ch) != TIEPIE_HW_TCM_NONE)
      {
        printf("    Condition               : ");
        printTriggerCondition(tiepie_hw_oscilloscope_channel_trigger_get_condition(scp, ch));
        printf("\n");
      }

      const uint32_t triggerTimeCount = tiepie_hw_oscilloscope_channel_trigger_get_time_count(scp, ch);
      if(triggerTimeCount > 0)
      {
        printf("    Times                   : ");
        for(uint32_t i = 0; i < triggerTimeCount; i++)
        {
          if(i != 0)
            printf(", ");

          printf("%f\n", tiepie_hw_oscilloscope_channel_trigger_get_time(scp, ch, i));
        }
        printf("\n");
      }
    }
  }

  printTriggerInputsInfo(scp);
  printTriggerOutputsInfo(scp);
}

void printGeneratorInfo(tiepie_hw_handle gen)
{
  if(gen == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid device handle in printGeneratorInfo()\n");
    return;
  }

  printf("Generator:\n");
  printf("  Connector type            : ");
  printConnectorType(tiepie_hw_generator_get_connector_type(gen));
  printf("\n");
  printf("  Differential              : %s\n", boolToStr(tiepie_hw_generator_is_differential(gen)));
  printf("  Controllable              : %s\n", boolToStr(tiepie_hw_generator_is_controllable(gen)));
  printf("  Impedance                 : %f\n", tiepie_hw_generator_get_impedance(gen));
  printf("  Resolution                : %" PRIu8 "\n", tiepie_hw_generator_get_resolution(gen));
  printf("  Output value min          : %f\n", tiepie_hw_generator_get_output_value_min(gen));
  printf("  Output value max          : %f\n", tiepie_hw_generator_get_output_value_max(gen));
  printf("  Output on                 : %s\n", boolToStr(tiepie_hw_generator_get_output_enable(gen)));
  if(tiepie_hw_generator_has_output_invert(gen))
  {
    printf("  Output invert             : %s\n", boolToStr(tiepie_hw_generator_get_output_invert(gen)));
  }

  printf("  Modes native              : ");
  printGeneratorMode(tiepie_hw_generator_get_modes_native(gen));
  printf("\n");
  printf("  Modes                     : ");
  printGeneratorMode(tiepie_hw_generator_get_modes(gen));
  printf("\n");
  if(tiepie_hw_generator_get_modes(gen) != TIEPIE_HW_GMM_NONE)
  {
    printf("  Burst mode                : ");
    printGeneratorMode(tiepie_hw_generator_get_mode(gen));
    printf("\n");
    if(tiepie_hw_generator_get_mode(gen) & TIEPIE_HW_GMM_BURST_COUNT)
    {
      printf("  Burst active              : %s\n", boolToStr(tiepie_hw_generator_is_burst_active(gen)));
      printf("  Burst count max           : %" PRIu64 "\n", tiepie_hw_generator_get_burst_count_max(gen));
      printf("  Burst count               : %" PRIu64 "\n", tiepie_hw_generator_get_burst_count(gen));
    }
    if(tiepie_hw_generator_get_mode(gen) & TIEPIE_HW_GMM_BURST_SAMPLE_COUNT)
    {
      printf("  Burst sample max          : %" PRIu64 "\n", tiepie_hw_generator_get_burst_sample_count_max(gen));
      printf("  Burst sample              : %" PRIu64 "\n", tiepie_hw_generator_get_burst_sample_count(gen));
    }
    if(tiepie_hw_generator_get_mode(gen) & TIEPIE_HW_GMM_BURST_SEGMENT_COUNT)
    {
      printf("  Burst segment max         : %" PRIu64 "\n", tiepie_hw_generator_get_burst_segment_count_max(gen));
      printf("  Burst segment             : %" PRIu64 "\n", tiepie_hw_generator_get_burst_segment_count(gen));
    }
  }

  printf("  Signal types              : ");
  printSignalType(tiepie_hw_generator_get_signal_types(gen));
  printf("\n");
  printf("  Signal type               : ");
  printSignalType(tiepie_hw_generator_get_signal_type(gen));
  printf("\n");

  if(tiepie_hw_generator_has_amplitude(gen))
  {
    printf("  Amplitude min             : %f\n", tiepie_hw_generator_get_amplitude_min(gen));
    printf("  Amplitude max             : %f\n", tiepie_hw_generator_get_amplitude_max(gen));
    printf("  Amplitude                 : %f\n", tiepie_hw_generator_get_amplitude(gen));

    uint32_t rangeCount = tiepie_hw_generator_get_amplitude_ranges(gen, NULL, 0);
    double ranges[sizeof(double) * rangeCount];
    rangeCount = tiepie_hw_generator_get_amplitude_ranges(gen, ranges, rangeCount);

    printf("  Amplitude ranges          : ");
    for(uint32_t i = 0; i < rangeCount; i++)
    {
      if(i != 0)
        printf(", ");

      printf("%f", ranges[i]);
    }
    printf("\n");

    printf("  Amplitude range           : %f\n", tiepie_hw_generator_get_amplitude_range(gen));
    printf("  Amplitude auto ranging    : %s\n", boolToStr(tiepie_hw_generator_get_amplitude_auto_ranging(gen)));
  }

  if(tiepie_hw_generator_has_frequency(gen))
  {
    printf("  Frequency modes           : ");
    printFrequencyMode(tiepie_hw_generator_get_frequency_modes(gen));
    printf("\n");
    printf("  Frequency mode            : ");
    printFrequencyMode(tiepie_hw_generator_get_frequency_mode(gen));
    printf("\n");
    printf("  Frequency min             : %f\n", tiepie_hw_generator_get_frequency_min(gen));
    printf("  Frequency max             : %f\n", tiepie_hw_generator_get_frequency_max(gen));
    printf("  Frequency                 : %f\n", tiepie_hw_generator_get_frequency(gen));
  }

  if(tiepie_hw_generator_has_offset(gen))
  {
    printf("  Offset min                : %f\n", tiepie_hw_generator_get_offset_min(gen));
    printf("  Offset max                : %f\n", tiepie_hw_generator_get_offset_max(gen));
    printf("  Offset                    : %f\n", tiepie_hw_generator_get_offset(gen));
  }

  if(tiepie_hw_generator_has_phase(gen))
  {
    printf("  Phase min                 : %f\n", tiepie_hw_generator_get_phase_min(gen));
    printf("  Phase max                 : %f\n", tiepie_hw_generator_get_phase_max(gen));
    printf("  Phase                     : %f\n", tiepie_hw_generator_get_phase(gen));
  }

  if(tiepie_hw_generator_has_symmetry(gen))
  {
    printf("  Symmetry min              : %f\n", tiepie_hw_generator_get_symmetry_min(gen));
    printf("  Symmetry max              : %f\n", tiepie_hw_generator_get_symmetry_max(gen));
    printf("  Symmetry                  : %f\n", tiepie_hw_generator_get_symmetry(gen));
  }

  if(tiepie_hw_generator_has_width(gen))
  {
    printf("  Width min                 : %f\n", tiepie_hw_generator_get_width_min(gen));
    printf("  Width max                 : %f\n", tiepie_hw_generator_get_width_max(gen));
    printf("  Width                     : %f\n", tiepie_hw_generator_get_width(gen));
  }

  if(tiepie_hw_generator_has_edge_time(gen))
  {
    printf("  Leading edge time min     : %f\n", tiepie_hw_generator_get_leading_edge_time_max(gen));
    printf("  Leading edge time max     : %f\n", tiepie_hw_generator_get_leading_edge_time_min(gen));
    printf("  Leading edge time         : %f\n", tiepie_hw_generator_get_leading_edge_time(gen));
    printf("  Trailing edge time min    : %f\n", tiepie_hw_generator_get_trailing_edge_time_max(gen));
    printf("  Trailing edge time max    : %f\n", tiepie_hw_generator_get_trailing_edge_time_min(gen));
    printf("  Trailing edge time        : %f\n", tiepie_hw_generator_get_trailing_edge_time(gen));
  }

  if(tiepie_hw_generator_has_data(gen))
  {
    printf("  DataLength min            : %" PRIu64 "\n", tiepie_hw_generator_get_data_length_min(gen));
    printf("  DataLength max            : %" PRIu64 "\n", tiepie_hw_generator_get_data_length_max(gen));
    printf("  DataLength                : %" PRIu64 "\n", tiepie_hw_generator_get_data_length(gen));
  }

  printTriggerInputsInfo(gen);
  printTriggerOutputsInfo(gen);
}

void printServerInfo(tiepie_hw_handle srv)
{
  if(srv == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid server handle in printServerInfo()\n");
    return;
  }

  printf("Server:\n");

  uint32_t length = tiepie_hw_server_get_url(srv, NULL, 0) + 1;
  char url[length];
  tiepie_hw_server_get_url(srv, url, length);
  printf("  URL                       : %s\n", url);

  length = tiepie_hw_server_get_name(srv, NULL, 0) + 1;
  char name[length];
  tiepie_hw_server_get_name(srv, name, length);
  printf("  Name                      : %s\n", name);

  length = tiepie_hw_server_get_description(srv, NULL, 0) + 1;
  char description[length];
  tiepie_hw_server_get_description(srv, description, length);
  printf("  Description               : %s\n", description);

  length = tiepie_hw_server_get_ip_address(srv, NULL, 0) + 1;
  char IPaddress[length];
  length = tiepie_hw_server_get_ip_address(srv, IPaddress, length);
  if(tiepie_hw_get_last_status() != TIEPIE_HW_STATUS_NOT_SUPPORTED)
  {
    printf("  IP address                : %s\n", IPaddress);
    printf("\n");
  }

  printf("  IP port                   : %" PRIu16 "\n", tiepie_hw_server_get_ip_port(srv));

  length = tiepie_hw_server_get_id(srv, NULL, 0) + 1;
  char id[length];
  tiepie_hw_server_get_id(srv, id, length);
  printf("  Id                        : %s\n", id);

  length = tiepie_hw_server_get_version(srv, NULL, 0) + 1;
  char version[length];
  tiepie_hw_server_get_version(srv, version, length);
  printf("  Version                   : %s\n", version);
  printf("\n");

  printf("  Status                    : %s\n", ServerStatuses[tiepie_hw_server_get_status(srv)]);

  if(tiepie_hw_server_get_last_error(srv) != TIEPIE_HW_SERVER_ERROR_NONE)
    printf("  Last error                : %s\n", ServerErrorCodes[tiepie_hw_server_get_last_error(srv)]);
}

void printTriggerInputsInfo(tiepie_hw_handle dev)
{
  if(dev == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid device handle in printTriggerInputsInfo()\n");
    return;
  }

  const uint16_t count = tiepie_hw_device_trigger_get_input_count(dev);

  for(uint16_t i = 0; i < count; i++)
  {
    printf("  TriggerInput %" PRIu16 ":\n", i);
    printf("    Id                      : %" PRIu32 "\n", tiepie_hw_device_trigger_input_get_id(dev, i));

    uint32_t length = tiepie_hw_device_trigger_input_get_name(dev, i, NULL, 0) + 1;
    char name[sizeof(char) * length];
    length = tiepie_hw_device_trigger_input_get_name(dev, i, name, length);
    printf("    Name                    : %s\n", name);

    printf("    Available               : %s\n", boolToStr(tiepie_hw_device_trigger_input_is_available(dev, i)));
    if(tiepie_hw_device_trigger_input_is_available(dev, i))
    {
      printf("    Enabled                 : %s\n", boolToStr(tiepie_hw_device_trigger_input_get_enabled(dev, i)));
      printf("    Kinds                   : ");
      printTriggerKind(tiepie_hw_device_trigger_input_get_kinds(dev, i));
      printf("\n");
      if(tiepie_hw_device_trigger_input_get_kinds(dev, i) != TIEPIE_HW_TKM_NONE)
      {
        printf("    Kind                    : ");
        printTriggerKind(tiepie_hw_device_trigger_input_get_kind(dev, i));
        printf("\n");
      }
    }
  }
}

void printTriggerOutputsInfo(tiepie_hw_handle dev)
{
  if(dev == TIEPIE_HW_HANDLE_INVALID)
  {
    fprintf(stderr, "Invalid device handle in printTriggerOutputsInfo()\n");
    return;
  }

  const uint16_t count = tiepie_hw_device_trigger_get_output_count(dev);

  for(uint16_t i = 0; i < count; i++)
  {
    printf("  TriggerOutput %" PRIu16 ":\n", i);
    printf("    Id                      : %" PRIu32 "\n", tiepie_hw_device_trigger_output_get_id(dev, i));

    uint32_t length = tiepie_hw_device_trigger_output_get_name(dev, i, NULL, 0) + 1;
    char name[sizeof(char) * length];
    length = tiepie_hw_device_trigger_output_get_name(dev, i, name, length);
    printf("    Name                    : %s\n", name);

    printf("    Enabled                 : %s\n", boolToStr(tiepie_hw_device_trigger_output_get_enabled(dev, i)));
    printf("    Events                  : ");
    printTriggerOutputEvent(tiepie_hw_device_trigger_output_get_events(dev, i));
    printf("\n");
    printf("    Event                   : ");
    printTriggerOutputEvent(tiepie_hw_device_trigger_output_get_event(dev, i));
    printf("\n");
  }
}

// Print functions for special values/types:
void printGeneratorMode(uint64_t generatorModes)
{
  if(generatorModes == TIEPIE_HW_GM_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_GMN_COUNT; i++)
  {
    if(generatorModes & (1ULL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", GeneratorModes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printClockOutput(uint32_t clockOutputs)
{
  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_CON_COUNT; i++)
  {
    if(clockOutputs & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", ClockOutputTypes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printClockSource(uint32_t clockSources)
{
  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_CSN_COUNT; i++)
  {
    if(clockSources & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", ClockSources[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printConnectorType(uint32_t connectorTypes)
{
  if(connectorTypes == TIEPIE_HW_CONNECTORTYPE_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_CONNECTORTYPE_COUNT; i++)
  {
    if(connectorTypes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", ConnectorTypes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printCoupling(uint64_t couplings)
{
  if(couplings == TIEPIE_HW_CK_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_CKN_COUNT; i++)
  {
    if(couplings & (1ULL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", Couplings[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printDate(tiepie_hw_date date)
{
  if(date.year != 0)
    printf("%04" PRIu16 "-%02" PRIu8 "-%02" PRIu8, (uint16_t)date.year, (uint8_t)date.month, (uint8_t)date.day);
  else
    printf("Unavailable");
}

void printDeviceType(uint32_t deviceTypes)
{
  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_DEVICETYPE_COUNT; i++)
  {
    if(deviceTypes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", DeviceTypes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printFrequencyMode(uint32_t frequencyModes)
{
  if(frequencyModes == TIEPIE_HW_FM_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_FMN_COUNT; i++)
  {
    if(frequencyModes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", FrequencyModes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printMeasureMode(uint32_t measureModes)
{
  if(measureModes == TIEPIE_HW_MM_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_MMN_COUNT; i++)
  {
    if(measureModes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", MeasureModes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printAutoResolutionMode(uint32_t autoResolutionModes)
{
  if(autoResolutionModes == TIEPIE_HW_ARMN_COUNT)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_ARMN_COUNT; i++)
  {
    if(autoResolutionModes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", AutoResolutionModes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printSignalType(uint32_t signalTypes)
{
  if(signalTypes == TIEPIE_HW_ST_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_STN_COUNT; i++)
  {
    if(signalTypes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", SignalTypes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printTriggerCondition(uint32_t triggerConditions)
{
  if(triggerConditions == TIEPIE_HW_TC_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_TCN_COUNT; i++)
  {
    if(triggerConditions & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", TriggerConditions[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printTriggerKind(uint64_t triggerKinds)
{
  if(triggerKinds == TIEPIE_HW_TK_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_TKN_COUNT; i++)
  {
    if(triggerKinds & (1ULL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", TriggerKinds[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printTriggerLevelMode(uint32_t triggerLevelModes)
{
  if(triggerLevelModes == TIEPIE_HW_TLM_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_TLMN_COUNT; i++)
  {
    if(triggerLevelModes & (1UL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", TriggerLevelModes[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printTriggerOutputEvent(uint64_t triggerOutputEvents)
{
  if(triggerOutputEvents == TIEPIE_HW_TOE_UNKNOWN)
  {
    printf("Unknown");
    return;
  }

  tiepie_hw_bool first = TIEPIE_HW_BOOL_TRUE;

  for(unsigned int i = 0; i < TIEPIE_HW_TOEN_COUNT; i++)
  {
    if(triggerOutputEvents & (1ULL << i))
    {
      if(!first)
        printf(", ");

      printf("%s", TriggerOutputEvents[i]);

      first = TIEPIE_HW_BOOL_FALSE;
    }
  }
}

void printVersion(const tiepie_hw_version* version)
{
  printf("%" PRIu16 ".%" PRIu16 ".%" PRIu16 ".%" PRIu16 "%s\n", version->major, version->minor, version->patch, version->build, version->extra);
}

// String conversion functions:
const char* boolToStr(tiepie_hw_bool value)
{
  return (value == TIEPIE_HW_BOOL_FALSE) ? "false" : "true";
}
