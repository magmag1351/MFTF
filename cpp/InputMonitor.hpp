#pragma once
#include <windows.h>
#include <mutex>
#include <thread>
#include <atomic>

class InputMonitor {
public:
    InputMonitor();
    ~InputMonitor();

    void start();
    void stop();
    void get_and_reset_counts(int& keys, int& clicks);

private:
    static LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam);
    static LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam);
    void loop();

    static HHOOK hKeyboardHook;
    static HHOOK hMouseHook;
    static std::atomic<int> key_count;
    static std::atomic<int> click_count;
    
    std::thread monitor_thread;
    std::atomic<bool> running;
    DWORD threadId;
};
