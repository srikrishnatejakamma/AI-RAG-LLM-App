package dev.rag.api;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ReactUiController {
    @GetMapping({"/react", "/react/"})
    public String app() {
        return "forward:/react/index.html";
    }
}
