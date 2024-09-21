#include <stdio.h>
#include <unistd.h>
#include <limits.h>

int main() {
    char cwd[PATH_MAX];

    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("v2 found the working directory for asmt2! %s\n", cwd);
    } else {
        perror("getcwd() error");
    }

    return 0;
}
