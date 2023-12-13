/**
 * PrintInfo.h
 *
 * This file is part of the LibTiePie programming examples.
 *
 * Find more information on https://www.tiepie.com/libtiepie-hw-sdk .
 */

#ifndef _PRINTINFO_H_
#define _PRINTINFO_H_

#include <libtiepie-hw.h>

// Print library info:
void printLibraryInfo();

// Print device info:
void printDeviceInfo(tiepie_hw_handle dev);

// Print oscilloscope info:
void printOscilloscopeInfo(tiepie_hw_handle scp);

// Print generator info:
void printGeneratorInfo(tiepie_hw_handle gen);

// Print server info:
void printServerInfo(tiepie_hw_handle srv);

// Print trigger input info:
void printTriggerInputsInfo(tiepie_hw_handle dev);

// Print trigger output info:
void printTriggerOutputsInfo(tiepie_hw_handle dev);

// Print functions for special values/types:
void printGeneratorMode(uint64_t generatorModes);
void printClockOutput(uint32_t clockOutputs);
void printClockSource(uint32_t clockSources);
void printConnectorType(uint32_t connectorTypes);
void printCoupling(uint64_t couplings);
void printDate(tiepie_hw_date date);
void printDeviceType(uint32_t deviceTypes);
void printFrequencyMode(uint32_t frequencyModes);
void printMeasureMode(uint32_t measureModes);
void printAutoResolutionMode(uint32_t autoResolutionModes);
void printSignalType(uint32_t signalTypes);
void printTriggerCondition(uint32_t triggerConditions);
void printTriggerKind(uint64_t triggerKinds);
void printTriggerLevelMode(uint32_t triggerLevelModes);
void printTriggerOutputEvent(uint64_t triggerOutputEvents);
void printVersion(const tiepie_hw_version* version);

// String conversion functions:
const char* boolToStr(tiepie_hw_bool value);

#endif
