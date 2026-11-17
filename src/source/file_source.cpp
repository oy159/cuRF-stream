#include "curf/sources.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace curf {

    FileSource::FileSource(FileConfig config) : config_(std::move(config)) {
        if (!config_.format.channels || !std::isfinite(config_.format.sample_rate) || config_.format.sample_rate <= 0)
            throw std::invalid_argument("FileSource: invalid stream format");
        file_.open(config_.path, std::ios::binary);
        if (!file_) throw std::runtime_error("FileSource: cannot open " + config_.path.string());
        const auto bytes = std::filesystem::file_size(config_.path);
        file_frames_ = bytes / config_.format.frame_bytes();
        if (file_frames_ == 0) throw std::runtime_error("FileSource: file holds no whole frame");
    }

    void FileSource::start() {
        next_sample_ = 0;
        file_.clear();
        t0_ = std::chrono::steady_clock::now();
    }

    ReadResult FileSource::read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) {
        const std::size_t fb = config_.format.frame_bytes();
        std::size_t want = dst.size() / fb;
        if (want == 0) return {};
        if (!config_.loop && next_sample_ >= file_frames_) return {ReadStatus::end_of_stream, 0, next_sample_, false};
        if (!config_.loop) want = std::min<std::size_t>(want, file_frames_ - next_sample_);
        if (config_.realtime) {
            using namespace std::chrono;
            if (t0_ == steady_clock::time_point{}) start();
            const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
            auto available = [&] {
                const auto produced = static_cast<std::uint64_t>(duration<double>(steady_clock::now() - t0_).count() * config_.format.sample_rate);
                return produced > next_sample_ ? produced - next_sample_ : 0;
            };
            while (available() < want && steady_clock::now() < deadline) std::this_thread::sleep_for(microseconds(200));
            want = std::min<std::size_t>(want, available());
            if (!want) return {ReadStatus::timeout, 0, next_sample_, false};
        }

        ReadResult r{ReadStatus::ok, 0, next_sample_, false};
        while (r.frames < want) {
            const std::uint64_t pos = next_sample_ % file_frames_;
            if (!config_.loop && next_sample_ >= file_frames_) break;
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(want - r.frames, file_frames_ - pos));
            file_.seekg(static_cast<std::streamoff>(pos * fb));
            file_.read(reinterpret_cast<char *>(dst.data() + r.frames * fb), static_cast<std::streamsize>(n * fb));
            if (!file_) throw std::runtime_error("FileSource: read failed");
            r.frames += n;
            next_sample_ += n;
        }
        if (r.frames == 0) r.status = ReadStatus::end_of_stream;
        return r;
    }

} // namespace curf
