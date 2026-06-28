#include "Arduino.h"
#include "SwitchControllerPico.h"

const uint8_t desc_hid_report[] = {
  CUSTOM_DESCRIPTOR
};

Adafruit_USBD_HID usb_hid(
  desc_hid_report,
  sizeof(desc_hid_report),
  HID_ITF_PROTOCOL_NONE,
  2,
  false
);

namespace {

constexpr unsigned long BAUD_RATE = 115200;
constexpr uint16_t BUTTON_PRESS_MS = 100;
constexpr uint16_t HAT_PRESS_MS = 260;
constexpr size_t MAX_COMMAND_LENGTH = 32;
constexpr uint8_t MAX_KEYS_PER_COMMAND = 3;
constexpr int UART_TX_PIN = 0;  // GP0: Pico -> USB serial adapter RX
constexpr int UART_RX_PIN = 1;  // GP1: USB serial adapter TX -> Pico
constexpr uint16_t MIN_HOLD_MS = 40;
constexpr uint16_t MAX_HOLD_MS = 1200;

String pending_command;

String normalizeCommand(String command) {
  command.trim();
  command.toUpperCase();
  return command;
}

bool sendButtonCommand(const String& command) {
  if (command == "A") {
    pushButton(Button::A, BUTTON_PRESS_MS, 1);
  } else if (command == "B") {
    pushButton(Button::B, BUTTON_PRESS_MS, 1);
  } else if (command == "X") {
    pushButton(Button::X, BUTTON_PRESS_MS, 1);
  } else if (command == "Y") {
    pushButton(Button::Y, BUTTON_PRESS_MS, 1);
  } else if (command == "START") {
    pushButton(Button::PLUS, BUTTON_PRESS_MS, 1);
  } else if (command == "SELECT") {
    pushButton(Button::MINUS, BUTTON_PRESS_MS, 1);
  } else {
    return false;
  }

  return true;
}

bool isButtonCommand(const String& command) {
  return command == "A" ||
         command == "B" ||
         command == "X" ||
         command == "Y" ||
         command == "START" ||
         command == "SELECT";
}

uint16_t clampHoldMs(long hold_ms) {
  if (hold_ms < MIN_HOLD_MS) {
    return MIN_HOLD_MS;
  }
  if (hold_ms > MAX_HOLD_MS) {
    return MAX_HOLD_MS;
  }
  return static_cast<uint16_t>(hold_ms);
}

bool parseTimedCommand(const String& command, String& base_command, uint16_t& hold_ms) {
  const int separator = command.indexOf(':');
  if (separator < 0) {
    base_command = command;
    return true;
  }

  base_command = command.substring(0, separator);
  const String hold_text = command.substring(separator + 1);
  if (base_command.length() == 0 || hold_text.length() == 0) {
    return false;
  }

  for (size_t i = 0; i < hold_text.length(); ++i) {
    if (!isDigit(hold_text[i])) {
      return false;
    }
  }

  hold_ms = clampHoldMs(hold_text.toInt());
  return true;
}

bool sendHatCommand(const String& command, uint16_t hold_ms) {
  if (command == "UP") {
    pushHatButton(Hat::UP, hold_ms, 1);
  } else if (command == "DOWN") {
    pushHatButton(Hat::DOWN, hold_ms, 1);
  } else if (command == "LEFT") {
    pushHatButton(Hat::LEFT, hold_ms, 1);
  } else if (command == "RIGHT") {
    pushHatButton(Hat::RIGHT, hold_ms, 1);
  } else {
    return false;
  }

  return true;
}

bool isHatCommand(const String& command) {
  return command == "UP" ||
         command == "DOWN" ||
         command == "LEFT" ||
         command == "RIGHT";
}

bool isValidCommand(const String& command) {
  String base_command;
  uint16_t hold_ms = HAT_PRESS_MS;
  if (!parseTimedCommand(command, base_command, hold_ms)) {
    return false;
  }
  return base_command == "WAIT" || isButtonCommand(base_command) || isHatCommand(base_command);
}

bool executeCommand(const String& command) {
  String base_command;
  uint16_t hold_ms = HAT_PRESS_MS;
  if (!parseTimedCommand(command, base_command, hold_ms)) {
    return false;
  }

  if (base_command == "WAIT") {
    delay(BUTTON_PRESS_MS);
    return true;
  }

  return sendButtonCommand(base_command) || sendHatCommand(base_command, hold_ms);
}

bool executeCommandList(const String& command, String& error) {
  int start = 0;
  uint8_t key_count = 0;
  String parts[MAX_KEYS_PER_COMMAND];

  while (start <= command.length()) {
    int separator = command.indexOf(',', start);
    if (separator < 0) {
      separator = command.length();
    }

    String part = command.substring(start, separator);
    part.trim();
    if (part.length() > 0) {
      ++key_count;
      if (key_count > MAX_KEYS_PER_COMMAND) {
        error = "too_many_keys";
        return false;
      }
      if (!isValidCommand(part)) {
        error = part;
        return false;
      }
      parts[key_count - 1] = part;
    }

    if (separator == command.length()) {
      break;
    }
    start = separator + 1;
  }

  if (key_count == 0) {
    error = "empty";
    return false;
  }

  for (uint8_t i = 0; i < key_count; ++i) {
    executeCommand(parts[i]);
  }
  return true;
}

void handleCommand(String raw_command) {
  const String command = normalizeCommand(raw_command);
  if (command.length() == 0) {
    return;
  }

  String error;
  if (executeCommandList(command, error)) {
    switchcontrollerpico_reset();
    Serial1.print("OK ");
    Serial1.println(command);
  } else {
    switchcontrollerpico_reset();
    Serial1.print("ERR ");
    Serial1.println(error.length() > 0 ? error : command);
  }
}

void readSerialCommands() {
  while (Serial1.available() > 0) {
    const char c = static_cast<char>(Serial1.read());

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      handleCommand(pending_command);
      pending_command = "";
      continue;
    }

    if (pending_command.length() < MAX_COMMAND_LENGTH) {
      pending_command += c;
    } else {
      pending_command = "";
      switchcontrollerpico_reset();
      Serial1.println("ERR command_too_long");
    }
  }
}

}  // namespace

void setup() {
  Serial1.setTX(UART_TX_PIN);
  Serial1.setRX(UART_RX_PIN);
  Serial1.begin(BAUD_RATE);

  switchcontrollerpico_init();

  while (!TinyUSBDevice.mounted()) {
    delay(1);
  }

  switchcontrollerpico_reset();
  Serial1.println("serial_hid_bridge ready");
}

void loop() {
  readSerialCommands();
}
