#include "curf/sources.hpp"

#include <algorithm>
#include <stdexcept>

namespace curf {

    FileSource::FileSource(FileConfig config) : config_(std::move(config)) {
        file_.open(config_.path, std::ios::binary);
        if (!file_) throw std::runtime_error("FileSource: cannot open " + config_.path.string());
        const auto bytes = std::filesystem::file_size(config_.path);
        file_frames_ = bytes / config_.format.frame_bytes();
        if (file_frames_ == 0) throw std::runtime_error("FileSource: file holds no whole frame");
    }

    ReadResult FileSource::read(std::span<std::uint8_t> dst, std::uint32_t) {
        const std::size_t fb = config_.format.frame_bytes();
        const std::size_t want = dst.size() / fb;
        if (want == 0) return {};

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
