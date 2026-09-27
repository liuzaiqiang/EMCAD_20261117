# Java 中的 HTTPS 完整理解与实践指南

> 面向已经写过 HTTP 接口、但还没有系统接触 HTTPS 的 Java 开发者。
>
> 本文目标是让你能回答三个问题：HTTPS 到底增加了什么、一次 HTTPS 请求具体发生了什么、在 Java 项目里怎样正确调用和提供 HTTPS 接口。

## 1. 先建立整体认识

### 1.1 一句话理解 HTTPS

HTTPS 可以理解为：**HTTP 运行在 TLS 加密通道之上**。

HTTP 负责定义应用层语义，例如：

- `GET /users/1` 表示读取资源；
- `POST /orders` 表示提交数据；
- 请求头中有 `Content-Type`、`Authorization`；
- 响应有状态码、响应头和响应体。

TLS 负责在 HTTP 发送之前建立一个安全通道，并提供：

1. **机密性**：旁观者难以直接看到请求和响应内容；
2. **完整性**：数据在传输途中被修改时能够被发现；
3. **身份认证**：客户端可以验证自己连接的服务器是否拥有可信证书；
4. **协商能力**：双方协商 TLS 版本、密码套件和应用层协议。

因此，HTTPS 不是另一套业务接口格式。你的 JSON、REST 路径、状态码和 Java Controller 仍然可以保持原来的写法，变化主要出现在连接建立、证书和部署配置上。

### 1.2 与 HTTP 的直观对比

| 项目 | HTTP | HTTPS |
|---|---|---|
| 默认端口 | 80 | 443 |
| 应用层格式 | HTTP | HTTP，通常是 HTTP/1.1 或 HTTP/2 |
| 传输保护 | 无 TLS 保护 | TLS 加密、校验完整性 |
| 是否需要证书 | 通常不需要 | 服务端通常需要证书 |
| URL 示例 | `http://api.example.com/users` | `https://api.example.com/users` |
| Java 主要客户端 | `HttpURLConnection`、Apache HttpClient、OkHttp、`java.net.http.HttpClient` | 同样的客户端，但会额外执行 TLS 握手和证书校验 |

### 1.3 HTTPS 不能解决什么

HTTPS 保护的是**传输链路**，不是所有安全问题：

- 服务器本身被入侵后，攻击者仍然可能读取业务数据；
- 服务器日志、数据库、消息队列中的敏感信息仍要单独保护；
- 如果客户端把证书校验关闭，HTTPS 的身份认证价值会被破坏；
- 用户输入、SQL 注入、越权、弱密码等应用层问题不会自动消失；
- HTTPS 不能保证接口一定可用，也不能替代鉴权、审计和限流。

## 2. 从一次请求的全链路看 HTTPS

假设 Java 客户端请求：

```text
https://api.example.com/users/1
```

实际过程可以拆成以下阶段：

1. **DNS 解析**：把 `api.example.com` 解析为 IP 地址；
2. **TCP 建立连接**：客户端与服务器的 443 端口建立 TCP 连接；
3. **TLS 握手**：协商版本和密码套件，服务器发送证书，双方建立会话密钥；
4. **HTTP 发送**：客户端在 TLS 加密通道中发送请求行、请求头和请求体；
5. **HTTP 响应**：服务器返回状态码、响应头和响应体；
6. **连接复用或关闭**：HTTP/1.1 Keep-Alive、HTTP/2 多路复用，或关闭连接。

可以把它画成：

```text
Java 应用
   ↓ HTTP 请求
HTTP
   ↓ 使用 TLS 保护
TLS
   ↓ 运行在 TCP 之上
TCP
   ↓
IP / 网络
```

服务器端则是反向过程。业务代码通常只看到 HTTP 层，JVM 和底层库负责 TLS 细节。

## 3. TLS 握手到底在做什么

### 3.1 握手的核心目标

握手不是简单地“把数据加密一下”，而是同时完成：

- 确认双方支持的 TLS 版本；
- 选择密码套件；
- 让服务器向客户端证明自己的身份；
- 生成本次连接专用的对称密钥；
- 确认后续握手消息没有被篡改。

### 3.2 TLS 1.3 的简化流程

现代 Java 运行环境通常优先使用 TLS 1.3，不能使用时再回退到 TLS 1.2。简化流程如下：

```text
客户端 → ClientHello
         支持的 TLS 版本、随机数、密码套件、密钥交换参数、SNI

服务器 → ServerHello
         选定版本、密码套件、服务器随机数、密钥交换参数
       → EncryptedExtensions
       → Certificate
         服务器证书链
       → CertificateVerify
         用服务器私钥证明自己拥有证书对应的私钥
       → Finished

客户端 → Finished
双方开始使用会话密钥加密 HTTP 数据
```

不同 TLS 版本的消息细节略有差异，但要抓住一个重点：**证书用于身份认证，临时密钥交换用于生成本次连接的对称密钥，真正大量传输的数据通常使用对称加密。**

### 3.3 为什么不直接用 RSA 加密全部 HTTP 数据

非对称密码算法适合解决身份认证和密钥交换，但直接加密大量数据速度较慢，且存在明文长度限制。HTTPS 的典型做法是：

1. 使用非对称密码或椭圆曲线密钥交换建立信任和共享秘密；
2. 从共享秘密派生出对称密钥；
3. 用速度更快的对称加密保护后续 HTTP 数据。

这也是为什么服务器证书中的公钥并不是“整个 HTTPS 通道的唯一加密钥匙”。

### 3.4 前向保密

TLS 1.2 和 TLS 1.3 常使用 ECDHE 等临时密钥交换。它的价值是：即使未来攻击者拿到了服务器长期私钥，也不应直接解开过去抓取的全部会话流量。每次连接使用临时密钥，可以降低长期密钥泄露造成的历史影响。

## 4. 证书、私钥和证书链

### 4.1 证书里有什么

X.509 证书通常包含：

- Subject：证书主体；
- Subject Alternative Name（SAN）：证书允许的域名或 IP；
- Issuer：签发者；
- Validity：生效时间和过期时间；
- Public Key：公钥；
- Key Usage / Extended Key Usage：密钥用途；
- Signature：上级 CA 对证书内容的签名。

现在校验域名时主要看 SAN。仅把域名写在旧式 CN 字段中，不能作为可靠的现代配置。

### 4.2 CA 是什么

CA（Certificate Authority，证书颁发机构）用自己的私钥签发证书。客户端通常预置一组可信根证书，形成如下信任链：

```text
根 CA（Root CA）
  └── 中间 CA（Intermediate CA）
        └── api.example.com 的服务器证书
```

客户端并不是因为“看到了一个证书”就信任服务器，而是要验证：

1. 证书仍在有效期内；
2. 请求的域名匹配 SAN；
3. 证书签名能沿链验证到本地信任库中的根 CA；
4. 证书用途允许作为服务器认证；
5. 证书没有明显的算法、密钥长度或策略问题。

### 4.3 服务端证书与客户端信任库

| 角色 | 持有内容 | 主要用途 |
|---|---|---|
| HTTPS 服务端 | 服务器证书 + 对应私钥 + 通常的中间证书链 | 向客户端证明服务器身份 |
| 普通 HTTPS 客户端 | truststore 中的可信根或中间证书 | 验证服务器证书 |
| 双向 TLS 客户端 | 客户端证书 + 私钥 | 向服务器证明客户端身份 |
| 双向 TLS 服务端 | 信任客户端证书的 truststore | 验证客户端身份 |

不要把 keystore 和 truststore 混为一谈：

- **keystore**：保存本方的私钥和证书，常用于“我是谁”；
- **truststore**：保存本方信任的 CA 或对端证书，常用于“我信谁”。

### 4.4 为什么访问 IP 可能失败

如果 URL 使用 `https://10.0.0.5`，但证书 SAN 只写了 `api.example.com`，域名校验会失败。解决办法是：

- 使用证书中包含的域名访问；
- 或重新签发包含该 IP 的证书；
- 测试环境可使用本地域名映射，但不能靠关闭校验绕过问题。

## 5. Java 中的 HTTPS 客户端

### 5.1 推荐的标准库：`java.net.http.HttpClient`

Java 11 及以上可以使用标准 HTTP Client。最简单的 HTTPS GET：

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class HttpsGetExample {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .build();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.example.com/users/1"))
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        System.out.println(response.statusCode());
        System.out.println(response.body());
    }
}
```

代码中没有显式写“加密算法”，因为默认 `SSLContext` 会由 JDK 使用系统安全配置、默认协议和默认信任库创建。生产代码应重点管理超时、连接复用、日志脱敏和异常分类。

### 5.2 设置连接和请求超时

```java
import java.time.Duration;

HttpClient client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(5))
        .version(HttpClient.Version.HTTP_2)
        .build();

HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("https://api.example.com/health"))
        .timeout(Duration.ofSeconds(10))
        .GET()
        .build();
```

连接超时表示建立连接阶段允许等待多久；请求超时表示整个请求等待时间。实际项目还要按业务拆分 DNS、连接、TLS 握手、读取和重试策略，避免把所有异常都无限重试。

### 5.3 使用 `SSLContext` 指定协议

```java
import javax.net.ssl.SSLContext;
import java.security.NoSuchAlgorithmException;

SSLContext sslContext = SSLContext.getInstance("TLS");
sslContext.init(null, null, null);

HttpClient client = HttpClient.newBuilder()
        .sslContext(sslContext)
        .build();
```

`TLS` 是协议族名称，具体启用哪些版本由 JDK 和安全策略决定。除非有明确兼容性需求，不要强行启用已经过时的 SSLv3、TLS 1.0 或 TLS 1.1。

### 5.4 使用自定义 truststore

当对接内部 CA 或测试环境自签名证书时，可以把可信证书放进专用 truststore，再构造 `SSLContext`：

```java
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyStore;

public final class SslContexts {
    public static SSLContext fromTrustStore(
            Path trustStorePath,
            char[] password
    ) throws Exception {
        KeyStore trustStore = KeyStore.getInstance("PKCS12");
        try (InputStream in = Files.newInputStream(trustStorePath)) {
            trustStore.load(in, password);
        }

        TrustManagerFactory factory = TrustManagerFactory.getInstance(
                TrustManagerFactory.getDefaultAlgorithm()
        );
        factory.init(trustStore);

        SSLContext context = SSLContext.getInstance("TLS");
        context.init(null, factory.getTrustManagers(), null);
        return context;
    }
}
```

调用示例：

```java
SSLContext sslContext = SslContexts.fromTrustStore(
        Path.of("config/internal-ca.p12"),
        System.getenv("TRUSTSTORE_PASSWORD").toCharArray()
);

HttpClient client = HttpClient.newBuilder()
        .sslContext(sslContext)
        .build();
```

密码不要硬编码在源码中。生产环境通常从密钥管理系统、环境变量或受保护的配置注入。

### 5.5 双向 TLS（mTLS）客户端

普通 HTTPS 主要认证服务端；mTLS 还要求客户端提供自己的证书。客户端需要同时配置：

1. `KeyManager`：从 keystore 读取客户端私钥和证书；
2. `TrustManager`：验证服务端证书；
3. `SSLContext`：把两者装配到 TLS 通道。

```java
KeyStore keyStore = KeyStore.getInstance("PKCS12");
try (InputStream in = Files.newInputStream(Path.of("client.p12"))) {
    keyStore.load(in, clientStorePassword);
}

KeyManagerFactory keyManagers = KeyManagerFactory.getInstance(
        KeyManagerFactory.getDefaultAlgorithm()
);
keyManagers.init(keyStore, clientKeyPassword);

KeyStore trustStore = KeyStore.getInstance("PKCS12");
try (InputStream in = Files.newInputStream(Path.of("server-ca.p12"))) {
    trustStore.load(in, serverTrustStorePassword);
}

TrustManagerFactory trustManagers = TrustManagerFactory.getInstance(
        TrustManagerFactory.getDefaultAlgorithm()
);
trustManagers.init(trustStore);

SSLContext context = SSLContext.getInstance("TLS");
context.init(
        keyManagers.getKeyManagers(),
        trustManagers.getTrustManagers(),
        null
);
```

## 6. Java 中提供 HTTPS 接口

### 6.1 Spring Boot 的常见配置

Spring Boot 内嵌 Tomcat、Jetty 或 Undertow 都可以监听 HTTPS。以 PKCS12 keystore 为例：

```yaml
server:
  port: 8443
  ssl:
    enabled: true
    key-store: classpath:server.p12
    key-store-type: PKCS12
    key-store-password: ${SERVER_KEYSTORE_PASSWORD}
    key-alias: server
    enabled-protocols: TLSv1.3,TLSv1.2
```

Controller 不需要因为 HTTPS 改成另一套写法：

```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    @GetMapping("/{id}")
    public User find(@PathVariable long id) {
        return userService.find(id);
    }
}
```

客户端使用 `https://host:8443/api/users/1` 即可访问。实际生产部署经常由 Nginx、Apache、云负载均衡或网关终止 TLS，再通过受控内网转发到 Java 服务。此时要正确配置转发头和安全边界。

### 6.2 HTTP 重定向到 HTTPS

常见部署方式是：

- 80 端口只负责把请求重定向到 HTTPS；
- 443 端口处理真正业务；
- API 客户端直接使用 HTTPS URL，避免依赖多次重定向。

重定向时要注意：敏感请求的 `POST`、认证头和请求体可能受客户端实现影响，不应把重定向当成安全控制的唯一手段。

### 6.3 TLS 终止在网关时的注意事项

如果浏览器或客户端到网关是 HTTPS、网关到 Java 是 HTTP，Java 应用看到的连接可能是普通 HTTP。需要明确：

- 网关是否可信且位于受保护网络；
- 内网是否也需要 HTTPS 或 mTLS；
- `X-Forwarded-Proto`、`Forwarded` 等头是否由可信网关设置；
- 应用是否正确生成 HTTPS 链接和 Cookie 的 `Secure` 属性；
- 是否防止外部客户端伪造转发头。

## 7. “为了能跑起来”最容易犯的错误

### 7.1 关闭证书校验

网上常见做法是自定义 `TrustManager`，无条件接受所有证书，或关闭 hostname verifier。这样可能让代码暂时访问成功，但攻击者可以伪造服务器或实施中间人攻击。

正确处理顺序是：

1. 检查证书是否过期；
2. 检查 URL 域名是否匹配 SAN；
3. 检查服务端是否返回完整中间证书链；
4. 确认客户端 JDK 信任库是否包含签发链；
5. 测试环境使用专用 truststore；
6. 只在隔离实验中临时验证问题，不把绕过代码带入生产。

### 7.2 只导入服务器叶子证书

导入单张叶子证书有时能解决眼前报错，但证书轮换后容易再次失败。更稳定的方式是信任内部根 CA，并由服务端正确发送中间证书链。

### 7.3 证书和私钥不匹配

服务端启动时常见错误是 keystore 中的证书与私钥不对应。应检查：

- alias 是否正确；
- keystore 密码与 key password 是否区分；
- 证书是否属于当前域名；
- 私钥是否仍然存在；
- PEM、JKS、PKCS12 格式是否被正确识别。

### 7.4 只在浏览器里验证

浏览器可能使用自己的缓存、系统证书库和代理配置。Java 使用的是 JVM 信任库、Java 安全策略和应用自己的代理设置。浏览器能访问，不代表 Java 一定能访问，应以 Java 实际日志和 JDK 工具检查为准。

## 8. 常见异常与定位方法

### 8.1 `PKIX path building failed`

典型含义是：Java 无法构建从服务器证书到可信根 CA 的证书链。

排查顺序：

1. 服务器是否发送了完整链；
2. 证书是否过期；
3. 使用的 JDK 是否包含对应根 CA；
4. 应用是否加载了预期 truststore；
5. 是否存在代理或中间设备替换证书；
6. 内部 CA 是否应导入专用 truststore。

### 8.2 `No subject alternative DNS name matching ...`

表示 URL 主机名和证书 SAN 不匹配。检查：

- URL 是否写错；
- 是否用 IP 访问了只包含域名的证书；
- 证书是否覆盖了新的域名、子域名或负载均衡地址。

### 8.3 `handshake_failure`

可能原因包括：

- TLS 版本没有交集；
- 密码套件没有交集；
- 服务端要求 mTLS，但客户端未提供证书；
- 客户端证书用途不正确；
- SNI 导致网关选择了错误的虚拟主机；
- 算法或密钥长度被安全策略拒绝。

### 8.4 `Connection reset` 或 `Read timed out`

它们不一定是证书问题，也可能是：

- 防火墙或代理主动关闭连接；
- 服务器负载过高；
- 连接池复用了失效连接；
- DNS、路由或 NAT 问题；
- 请求超时设置太短。

先区分 DNS、TCP、TLS、HTTP 四个阶段，避免把所有网络异常都归因于 HTTPS。

## 9. 调试与证据收集

### 9.1 查看 Java TLS 握手日志

可以临时使用 JVM 参数：

```text
-Djavax.net.debug=ssl,handshake
```

它会输出大量握手信息，适合在测试环境复现问题。生产环境谨慎开启，避免日志过大或泄露证书、主机名等信息。

### 9.2 使用 `keytool` 查看证书

```bash
keytool -list -v -keystore server.p12 -storetype PKCS12
keytool -list -v -keystore truststore.p12 -storetype PKCS12
```

重点看：证书链长度、有效期、SAN、Extended Key Usage、alias，以及是否确实加载了目标文件。

### 9.3 检查服务端证书链

可以使用 OpenSSL 查看远端链：

```bash
openssl s_client -connect api.example.com:443 \
  -servername api.example.com -showcerts
```

`-servername` 很重要，因为同一个 IP 可能承载多个域名。拿到输出后检查叶子证书、Issuer、有效期和中间证书是否完整。

### 9.4 最小化复现原则

当 Java 项目访问失败时，建议建立一个最小复现程序，只保留：

- 一个 URL；
- 一个 HttpClient；
- 明确的 JDK 版本；
- 明确的 truststore 路径；
- 完整异常堆栈；
- TLS 调试日志。

这样可以判断问题来自证书、JDK、代理，还是业务框架的连接池和配置。

## 10. 生产环境设计清单

### 10.1 客户端

- 使用 HTTPS URL，禁止敏感接口回退到 HTTP；
- 保留默认证书链校验和主机名校验；
- 为连接、请求和读取设置合理超时；
- 使用连接池和复用，避免每次请求都重新握手；
- 对重试设置上限，并区分幂等请求与非幂等请求；
- 不记录 Authorization、Cookie、密码、身份证号和完整响应体；
- 内部 CA 使用受控 truststore，并建立证书轮换流程；
- mTLS 私钥放在安全的密钥存储中；
- 记录 TLS 失败的阶段和错误类型，而不是只记录“请求失败”。

### 10.2 服务端

- 使用受支持的 TLS 版本；
- 使用合法、未过期、域名匹配的证书；
- 正确部署中间证书链；
- 私钥权限最小化，禁止提交到 Git；
- 配置证书到期监控和自动轮换；
- 清楚定义 TLS 终止点；
- 对外部请求设置安全响应头、限流和审计；
- 若使用 mTLS，明确客户端证书签发、吊销和轮换流程。

## 11. 从 HTTP 迁移到 HTTPS 的实践步骤

### 第一步：先验证服务端

确认域名、端口、证书 SAN、证书链和过期时间。先用浏览器或 OpenSSL 检查，再用 Java 客户端验证。

### 第二步：确认 JDK 和依赖

记录 Java 版本、厂商、操作系统、HTTP 客户端版本和运行参数。不同 JDK 的默认根证书、TLS 支持和安全策略可能不同。

### 第三步：先写普通单向 HTTPS

先使用默认 truststore 访问公有 CA 签发的服务，不要一开始就引入自定义 TrustManager。确认最基本链路成功后，再加入内部 CA 或 mTLS。

### 第四步：补齐超时、日志和重试

HTTPS 可能增加握手耗时，但不能靠无限延长超时解决问题。记录请求阶段、耗时、状态码和经过脱敏的错误信息。

### 第五步：做证书轮换演练

在测试环境更换服务端证书、中间证书和客户端 truststore，确认应用不需要临时改代码或关闭校验。

## 12. 常见概念辨析

### 12.1 HTTPS、SSL、TLS

SSL 是较早的安全协议名称，现代 HTTPS 应使用 TLS。很多配置项仍然写着 `ssl`，例如 `SSLContext`、`javax.net.ssl`，这是历史命名，不代表必须使用旧版 SSL。

### 12.2 加密、编码、哈希、签名

| 概念 | 是否可逆 | 主要用途 |
|---|---|---|
| 加密 | 有密钥时可逆 | 保密 |
| Base64 编码 | 可逆且无密钥 | 表示二进制数据，不提供安全性 |
| 哈希 | 通常不可逆 | 完整性、索引、口令存储方案的一部分 |
| 数字签名 | 可验证，不能还原原文 | 身份认证、完整性、不可否认性 |

HTTPS 同时使用这些机制中的一部分，但它们不是同一个概念。

### 12.3 证书过期与证书不受信

证书过期表示时间校验失败；不受信表示证书链无法连接到信任根。两者修复方式不同：前者通常需要续期，后者需要修复链或正确配置 truststore。

### 12.4 HTTP 认证与 HTTPS 认证

`Authorization: Bearer ...`、Basic Auth、OAuth2 等是应用层身份认证；服务器证书认证是 TLS 层身份认证。实际接口通常需要两层都正确：TLS 确认连接的是可信服务器，应用认证确认用户或服务是否有权限。

## 13. 一个适合学习的实验路线

### 实验一：观察普通 HTTPS 请求

使用 Java 11+ 的 `HttpClient` 访问一个公开 HTTPS 地址，打印状态码、响应头和响应体，确认业务代码与 HTTP 基本一致。

### 实验二：故意使用错误域名

把 URL 改成证书不包含的域名，观察主机名校验异常，理解 SAN 的作用。

### 实验三：使用自签名证书

在本地服务生成测试证书，把 CA 导入专用 truststore，使用自定义 `SSLContext` 访问。实验结束后删除绕过校验的代码。

### 实验四：双向 TLS

为服务端和客户端分别生成证书，服务端只信任指定客户端 CA，观察没有客户端证书时的握手失败，再配置 `KeyManager` 使其成功。

### 实验五：证书轮换

替换服务端证书或中间 CA，验证客户端是否依赖单张叶子证书，记录轮换前后的行为。

## 14. 学习时应形成的心智模型

建议把 HTTPS 问题按四层定位：

```text
应用层：URL、HTTP 方法、状态码、JSON、鉴权
TLS 层：证书、SAN、信任链、协议、密码套件、mTLS
传输层：TCP 连接、端口、超时、连接复用
网络层：DNS、路由、防火墙、代理、负载均衡
```

看到异常时先问：

1. DNS 是否解析成功？
2. TCP 是否连上目标端口？
3. TLS 是否完成握手？
4. HTTP 请求是否真正发出？
5. 服务端返回了什么状态码？
6. 是否存在代理、网关或证书替换？

这样可以避免把业务 401、证书校验失败和 TCP 超时混为一谈。

## 15. 快速记忆版

如果只记住一页内容，可以记住以下结论：

1. HTTPS = HTTP + TLS；
2. TLS 负责机密性、完整性和身份认证；
3. 证书证明服务器是谁，私钥证明服务器确实拥有证书对应的秘密；
4. 证书链必须能连接到客户端信任库中的根 CA；
5. 域名校验主要看证书 SAN；
6. 大量数据通常使用对称密钥加密，证书公钥主要参与身份认证和密钥协商；
7. Java 默认会做证书和主机名校验，生产环境不要关闭；
8. `PKIX path building failed` 优先检查信任链和 truststore；
9. `No subject alternative ...` 优先检查 URL 与 SAN；
10. HTTPS 不能替代鉴权、输入校验、日志保护和权限控制。

## 16. 参考资料

- RFC 8446：The Transport Layer Security (TLS) Protocol Version 1.3：<https://www.rfc-editor.org/rfc/rfc8446>
- RFC 5280：Internet X.509 Public Key Infrastructure Certificate and CRL Profile：<https://www.rfc-editor.org/rfc/rfc5280>
- Oracle Java Secure Socket Extension（JSSE）参考指南：<https://docs.oracle.com/en/java/javase/21/security/java-secure-socket-extension-jsse-reference-guide.html>
- Oracle `java.net.http` 包文档：<https://docs.oracle.com/en/java/javase/21/docs/api/java.net.http/module-summary.html>
- Spring Boot SSL 配置文档：<https://docs.spring.io/spring-boot/reference/features/ssl.html>
- OpenSSL `s_client` 文档：<https://docs.openssl.org/>

> 参考资料用于进一步核对协议、JDK API 和框架配置。实际部署时仍应以当前使用的 JDK、Spring Boot、反向代理和证书服务商文档为准。

